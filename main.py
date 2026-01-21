import os
import shutil
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cuda 
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from rdkit import Chem
from rdkit import RDLogger
from lifelines.utils import concordance_index
import gc

# ================= 🚀 A100 DAVIS 重训配置 =================
DATASET_NAME = 'davis'
BATCH_SIZE = 256        # A100 显存大，直接 256
EPOCHS = 200            # 一步到位
LR = 0.0001             # 初始学习率 (因为是从头学，所以用 0.0005)
ALPHA = 0.01            # 极低对比损失权重，主攻 MSE
SAVE_DIR = './checkpoints' 
# ========================================================

RDLogger.DisableLog('rdApp.*') 
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# 显卡加速
torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# 1. 准备数据
def prepare_data():
    data_root = './data'
    target_file = os.path.join(data_root, DATASET_NAME, 'ligands_can.txt')
    if not os.path.exists(target_file):
        print(f"📦 下载数据中...")
        if os.path.exists("DeepDTA"): shutil.rmtree("DeepDTA")
        os.system("git clone https://mirror.ghproxy.com/https://github.com/hkmztrk/DeepDTA.git")
        os.makedirs(data_root, exist_ok=True)
        shutil.move(f"DeepDTA/data/{DATASET_NAME}", data_root)
        if os.path.exists("DeepDTA"): shutil.rmtree("DeepDTA")

# 2. 特征工具
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()], dtype=np.float32)

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    if len(edges) == 0: 
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.tensor(features, dtype=torch.float), edge_index

# 3. Dataset
class GeneralDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, drugs, prots, y, transform=None, pre_transform=None):
        self.dataset_name = dataset_name
        self.drugs = drugs
        self.prots = prots
        self.y = y
        super().__init__(root, transform, pre_transform)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self): return [f'{self.dataset_name}_processed.pt']

    def process(self):
        data_list = []
        CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25 }
        print(f"🔨 处理数据中...")
        for i in range(len(self.drugs)):
            x, edge_index = smile_to_graph(self.drugs[i])
            if x is None: continue
            target = [CHARPROTSET.get(c, 0) for c in self.prots[i]]
            if len(target) > 1000: target = target[:1000]
            else: target = target + [0]*(1000-len(target))
            data = Data(x=x, edge_index=edge_index, 
                        target=torch.tensor(target, dtype=torch.long).unsqueeze(0), 
                        y=torch.tensor([self.y[i]], dtype=torch.float))
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"✅ 数据处理完成!")

# 4. Model
class CL_DTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_conv1 = GATv2Conv(78, 128, heads=4, concat=False, dropout=0.1)
        self.drug_conv2 = GATv2Conv(128, 128, heads=4, concat=False, dropout=0.1)
        self.drug_conv3 = GATv2Conv(128, 128, heads=4, concat=False, dropout=0.1)
        encoder_layer = TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True, dropout=0.1)
        self.prot_trans = TransformerEncoder(encoder_layer, num_layers=2)
        self.prot_embed = nn.Embedding(26, 128)
        self.prot_fc = nn.Linear(128, 128)
        self.regressor = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
        self.drug_proj = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64))
        self.prot_proj = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64))

    def forward(self, data):
        x = F.elu(self.drug_conv1(data.x, data.edge_index))
        x = F.elu(self.drug_conv2(x, data.edge_index))
        x = self.drug_conv3(x, data.edge_index)
        d_emb = global_mean_pool(x, data.batch)
        p = self.prot_embed(data.target)
        p = self.prot_trans(p)
        p_emb = self.prot_fc(p.mean(dim=1))
        pred = self.regressor(torch.cat([d_emb, p_emb], dim=1))
        z_d = self.drug_proj(d_emb)
        z_p = self.prot_proj(p_emb)
        return pred, z_d, z_p

def contrastive_loss(z_i, z_j, temp=0.1):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    logits = torch.matmul(z_i, z_j.T) / temp
    labels = torch.arange(z_i.size(0)).to(z_i.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

# 5. 训练主程序
def run_restart():
    gc.collect()
    torch.cuda.empty_cache()
    prepare_data()
    
    print(f"\n{'='*40}")
    print(f"🚀 A100 DAVIS 重生之战 (ALPHA={ALPHA})")
    print(f"🔥 策略: 从零开始，Log转换 + Z-Score标准化")
    print(f"{'='*40}")
    
    # 强制清理缓存，确保使用最新的标准化逻辑
    processed_path = './data/processed_optimal_davis'
    if os.path.exists(processed_path): shutil.rmtree(processed_path)
    
    data_path = f'./data/{DATASET_NAME}/'
    ligands = json.load(open(data_path + 'ligands_can.txt'))
    proteins = json.load(open(data_path + 'proteins.txt'))
    Y = pickle.load(open(data_path + 'Y', 'rb'), encoding='latin1')
    
    try:
        drug_keys = sorted(ligands.keys(), key=lambda x: int(x))
        prot_keys = sorted(proteins.keys(), key=lambda x: int(x))
    except:
        drug_keys = sorted(ligands.keys())
        prot_keys = sorted(proteins.keys())

    drugs_list = [ligands[k] for k in drug_keys]
    prots_list = [proteins[k] for k in prot_keys]
    
    drugs, prots, affinities = [], [], []
    raw_y_sample = []
    
    for i in range(len(drugs_list)):
        for j in range(len(prots_list)):
            try:
                val = Y[i][j]
                if not np.isnan(val):
                    drugs.append(drugs_list[i])
                    prots.append(prots_list[j])
                    affinities.append(val)
                    if len(raw_y_sample) < 100: raw_y_sample.append(val)
            except: continue
            
    affinities = np.array(affinities)
    
    # 1. Log 转换
    if np.mean(raw_y_sample) > 100: 
        print(f"⚠️ 执行 -Log10 转换...")
        affinities = -np.log10(affinities / 1e9)
        
    # 2. Z-Score 标准化 (这是之前缺失的关键！)
    Y_mean = np.mean(affinities)
    Y_std = np.std(affinities)
    affinities_norm = (affinities - Y_mean) / Y_std
    
    print(f"📊 统计: Mean={Y_mean:.4f}, Std={Y_std:.4f}")
    print(f"   标准化后: Mean={np.mean(affinities_norm):.4f}, Std={np.std(affinities_norm):.4f}")
    
    dataset = GeneralDataset(root=processed_path, dataset_name=DATASET_NAME, 
                             drugs=drugs, prots=prots, y=affinities_norm)
    
    train_size = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    
    device = torch.device('cuda')
    model = CL_DTA().to(device) # 全新初始化的模型
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    mse_crit = nn.MSELoss()
    
    best_ci = -1
    
    print("\n⚡️ 训练开始...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred, z_d, z_p = model(batch)
            loss = mse_crit(pred.flatten(), batch.y) + ALPHA * contrastive_loss(z_d, z_p)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device, non_blocking=True)
                p, _, _ = model(batch)
                # 还原真实值
                p_real = p * Y_std + Y_mean
                y_real = batch.y * Y_std + Y_mean
                preds.extend(p_real.cpu().numpy().flatten())
                targets.extend(y_real.cpu().numpy().flatten())
        
        ci = concordance_index(targets, preds)
        mse = np.mean((np.array(preds) - np.array(targets))**2)
        scheduler.step(mse)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | MSE: {mse:.4f} | CI: {ci:.4f} | LR: {current_lr:.1e}")
        
        if ci > best_ci:
            best_ci = ci
            torch.save(model.state_dict(), f'{SAVE_DIR}/{DATASET_NAME}_optimal.pth')
            # 保存统计数据以便后续使用
            with open(f'{SAVE_DIR}/{DATASET_NAME}_stats.pkl', 'wb') as f:
                pickle.dump({'mean': Y_mean, 'std': Y_std}, f)
            
    print(f"🏆 训练结束! 最佳 CI: {best_ci:.4f}")

if __name__ == "__main__":
    run_restart()
