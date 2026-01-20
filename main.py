# ==============================================================================
# CL-GraphTrans-DTA (通用算力云 最终修正版)
# ==============================================================================
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

# ================= 配置区域 (在这里修改参数) =================
DATASET_NAME = 'davis'  # 想跑 KIBA 就改成 'kiba'
BATCH_SIZE = 128        # 显存 > 24G (如3090/4090) 可改为 256 或 512
EPOCHS = 100
LR = 0.0005
SAVE_DIR = './checkpoints' # 模型保存路径
# ============================================================

# --- 0. 全局设置 ---
RDLogger.DisableLog('rdApp.*') # 屏蔽 RDKit 警告
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 显卡加速优化
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# --- 1. 数据下载与准备 ---
def prepare_data():
    data_root = './data'
    target_file = os.path.join(data_root, DATASET_NAME, 'ligands_can.txt')
    
    if not os.path.exists(target_file):
        print(f"📦 未检测到 {DATASET_NAME} 数据，正在下载...")
        # 清理残余
        if os.path.exists("DeepDTA"): shutil.rmtree("DeepDTA")
        
        # 使用 GitHub 代理加速下载 (适合国内云服务器)
        # 如果下载失败，请尝试去掉 'https://mirror.ghproxy.com/'
        clone_cmd = "git clone https://mirror.ghproxy.com/https://github.com/hkmztrk/DeepDTA.git"
        os.system(clone_cmd)
        
        os.makedirs(data_root, exist_ok=True)
        
        source_dir = f"DeepDTA/data/{DATASET_NAME}"
        target_dir = f"{data_root}/{DATASET_NAME}"
        
        # 移动数据
        if os.path.exists(source_dir):
            if os.path.exists(target_dir): shutil.rmtree(target_dir)
            shutil.move(source_dir, data_root)
            print(f"✅ {DATASET_NAME} 下载并移动完成！")
        else:
            print("❌ 下载失败或路径错误，请检查网络或手动上传数据。")
        
        # 清理
        if os.path.exists("DeepDTA"): shutil.rmtree("DeepDTA")
    else:
        print(f"✅ 检测到本地 {DATASET_NAME} 数据完整。")

# --- 2. 分子图转换工具 ---
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

# --- 3. 数据集类 (已修复 weights_only 报错) ---
class GeneralDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, drugs, prots, y, transform=None, pre_transform=None):
        self.dataset_name = dataset_name
        self.drugs = drugs
        self.prots = prots
        self.y = y
        super().__init__(root, transform, pre_transform)
        
        # 🛠️ 关键修复：显式设置 weights_only=False 兼容 PyTorch 2.6+
        # PyG 的 Data 对象属于复杂对象，必须允许非纯权重加载
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError: # 兼容旧版 PyTorch (没有 weights_only 参数的情况)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self): return [f'{self.dataset_name}_processed.pt']

    def process(self):
        data_list = []
        CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25 }
        
        print(f"🔨 正在处理 {self.dataset_name} 数据 (初次运行需要几分钟)...")
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
        print(f"✅ {self.dataset_name} 处理并缓存完成!")

# --- 4. 模型结构 ---
class CL_DTA(nn.Module):
    def __init__(self):
        super().__init__()
        # GATv2 用于分子图
        self.drug_conv1 = GATv2Conv(78, 128, heads=4, concat=False, dropout=0.1)
        self.drug_conv2 = GATv2Conv(128, 128, heads=4, concat=False, dropout=0.1)
        self.drug_conv3 = GATv2Conv(128, 128, heads=4, concat=False, dropout=0.1)
        
        # Transformer 用于蛋白质序列
        encoder_layer = TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True, dropout=0.1)
        self.prot_trans = TransformerEncoder(encoder_layer, num_layers=2)
        self.prot_embed = nn.Embedding(26, 128)
        self.prot_fc = nn.Linear(128, 128)
        
        # 预测头与投影头
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

# --- 5. 训练逻辑 ---
def contrastive_loss(z_i, z_j, temp=0.1):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    logits = torch.matmul(z_i, z_j.T) / temp
    labels = torch.arange(z_i.size(0)).to(z_i.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

def run():
    # 强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    
    # 1. 准备数据
    prepare_data()
    print(f"\n🚀 开始运行 {DATASET_NAME.upper()} (Batch Size={BATCH_SIZE})")
    
    # 2. 清理旧缓存 (非常重要：防止 weights_only 报错后残留损坏文件)
    processed_path = './data/processed'
    if os.path.exists(processed_path): 
        print("🧹 清理旧缓存，准备重新生成数据...")
        shutil.rmtree(processed_path)
    
    # 3. 读取原始文件并处理
    data_path = f'./data/{DATASET_NAME}/'
    ligands = json.load(open(data_path + 'ligands_can.txt'))
    proteins = json.load(open(data_path + 'proteins.txt'))
    Y = pickle.load(open(data_path + 'Y', 'rb'), encoding='latin1')
    
    # 智能排序 Key (防止 KeyError '0')
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
    
    print("正在对齐数据...")
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
            
    # 数值转换检测 (Log转换)
    affinities = np.array(affinities)
    if np.mean(raw_y_sample) > 10: 
        print(f"⚠️ 检测到原始值 (Mean: {np.mean(raw_y_sample):.2f})，执行 Log 转换...")
        affinities = -np.log10(affinities / 1e9)
        print(f"✅ 转换完成。范围: {affinities.min():.2f} - {affinities.max():.2f}")
    
    print(f"样本数量: {len(affinities)}")
    
    # 4. 创建 Dataset (这里会调用修复后的 GeneralDataset)
    dataset = GeneralDataset(root=processed_path, dataset_name=DATASET_NAME, 
                             drugs=drugs, prots=prots, y=affinities)
    
    train_size = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    # 5. DataLoader (多线程加速)
    num_workers = 8 # 云服务器建议开 8-16
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    # 6. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    model = CL_DTA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_crit = nn.MSELoss()
    
    best_ci = -1
    print("\n⚡️ 训练开始...")
    
    # 7. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred, z_d, z_p = model(batch)
            loss = mse_crit(pred.flatten(), batch.y) + 0.2 * contrastive_loss(z_d, z_p)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 验证
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                p, _, _ = model(batch)
                preds.extend(p.cpu().numpy().flatten())
                targets.extend(batch.y.cpu().numpy().flatten())
        
        ci = concordance_index(targets, preds)
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Test CI: {ci:.4f}")
        
        if ci > best_ci:
            best_ci = ci
            save_path = os.path.join(SAVE_DIR, f'{DATASET_NAME}_best.pth')
            torch.save(model.state_dict(), save_path)
            
    print(f"🏆 {DATASET_NAME} 实验结束! 最佳 CI: {best_ci:.4f}，模型已保存至 {SAVE_DIR}")

if __name__ == "__main__":
    run()
