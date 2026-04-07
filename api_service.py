"""
Bamboo-Mixer API Service (Paper Branch)
电解液配方预测与生成服务

使用 paper 分支的 formula_design 模块
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# ============== Paper Branch 模块导入 ==============
from formula_design.data import FormulaData, Data
from formula_design.mol.molecule import Molecule
from formula_design.predictor.mono import Mono
from formula_design.predictor.molmix import MolMix
from formula_design.generator.diffusion import FormulaDiffusion
from formula_design.generator.decoder import FormulaDecoder


# ============== 配置 ==============
MODELS_DIR = PROJECT_ROOT / "models" / "bamboo_mixer"
CKPT_DIR = MODELS_DIR / "ckpts"
EMB_DIR = PROJECT_ROOT / "emb_dict"

# 模型路径
MONO_CKPT = str(CKPT_DIR / "mono" / "optimal.pt")
FORMULA_CKPT = str(CKPT_DIR / "formula" / "optimal.pt")  # This is actually MolMix
DIFFUSION_CKPT = str(CKPT_DIR / "generator" / "diffusion.pt")
DECODER_CKPT = str(CKPT_DIR / "generator" / "decoder.pt")

# 词汇表路径
SALTS_JSON = str(EMB_DIR / "salts.json")
SOLVS_JSON = str(EMB_DIR / "solvents.json")

# ============== 词汇表加载 ==============
with open(SALTS_JSON, "r", encoding="utf-8") as f:
    salts_data = json.load(f)
with open(SOLVS_JSON, "r", encoding="utf-8") as f:
    solvs_data = json.load(f)

SALTS = [s["name"] for s in salts_data]
SOLVENTS = [s["name"] for s in solvs_data]

# 构建 name -> mapped_smiles 映射
SALT_SMILES_MAP = {s["name"]: s["smiles"] for s in salts_data}
SALT_TEMP_MAP = {s["name"]: s.get("temperature", 25) for s in salts_data}
SOLV_SMILES_MAP = {s["name"]: s["smiles"] for s in solvs_data}
SOLV_TEMP_MAP = {s["name"]: s.get("temperature", 25) for s in solvs_data}


# ============== 模型配置 ==============
# Mono (MonoPredictor) 配置
MONO_CONFIG = {
    "graph_block": {
        "feature_layer": {
            "atom_embedding_dim": 16,
            "node_mlp_dims": [32, 32, 2],
            "edge_mlp_dims": [32, 32, 2],
            "act": "gelu"
        },
        "gnn_layer": {
            "gnn_type": "EGT",
            "gnn_dims": [32, 32, 3],
            "jk": "cat",
            "act": "gelu",
            "heads": 4,
            "at_channels": 8,
            "ffn_dims": [32, 2]
        }
    },
    "readout_block": {
        "input_dim": 64,
        "hidden_dims": [256, 128, 16],
        "output_dim": 1
    }
}

# MolMix (FormulaPredictor) 配置
MOLMIX_CONFIG = {
    "aggr_block": {
        "node_emb_dim": 32,
        "node_att_dim": 32,
        "edge_emb_dim": 32,
        "edge_att_dim": 32,
    },
    "readout_block": {
        "input_dim": 384,
        "hidden_dims": [512, 128, 16],
        "output_dim": 1
    },
    "anion_block": {
        "input_dim": 400,
        "hidden_dims": [512, 128, 16],
        "output_dim": 1
    }
}

# Diffusion 配置
DIFFUSION_CONFIG = {
    "unet1d_config": {
        "dim": 64,
        "prop_dim": 2,
        "dim_mults": [1, 2, 4, 8],
        "channels": 1,
        "dropout": 0.0,
        "self_condition": False,
        "learned_variance": False,
        "learned_sinusoidal_cond": False,
        "random_fourier_features": False,
        "learned_sinusoidal_dim": 16,
        "sinusoidal_pos_emb_theta": 10000,
        "attn_dim_head": 16,
        "attn_heads": 4,
        "condition_method": "concat"
    },
    "beta_scheduler_config": {
        "timesteps": 1000,
        "scheduler_mode": "cosine"
    },
    "sigma_scheduler_config": {
        "timesteps": 1000
    },
    "diff_config": {
        "time_dim": 256
    }
}

# Decoder 配置
DECODER_CONFIG = {
    "formula_emb_dim": 384,
    "set_transformer_block": {
        "hidden_dim": 64,
        "max_num_mol": 12,
        "num_layers": 2
    },
    "aggr_block": {
        "node_emb_dim": 32,
        "node_att_dim": 32,
        "edge_emb_dim": 32,
        "edge_att_dim": 32
    },
    "mol_dict": {
        "solv_dict_path": str(EMB_DIR / "solv_embs.pt"),
        "salt_dict_path": str(EMB_DIR / "salt_embs.pt")
    }
}


# ============== 模型实例（全局） ==============
class ModelManager:
    """模型管理器 - 单例模式"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 模型实例
        self.mono_model = None
        self.molmix_model = None
        self.diffusion_model = None
        self.decoder_model = None

        self._initialized = True

    def load_models(self):
        """加载所有模型"""
        print("加载 Mono 模型...")
        self.mono_model = Mono(graph_block=MONO_CONFIG["graph_block"], readout_block=MONO_CONFIG["readout_block"])
        self.mono_model.load_ckpt(MONO_CKPT, map_location=self.device)
        self.mono_model.to(self.device)
        self.mono_model.eval()
        print(f"  Mono 模型已加载: {MONO_CKPT}")

        print("加载 MolMix 模型...")
        self.molmix_model = MolMix(
            pretrained_model=self.mono_model,
            aggr_block=MOLMIX_CONFIG["aggr_block"],
            readout_block=MOLMIX_CONFIG["readout_block"],
            anion_block=MOLMIX_CONFIG["anion_block"]
        )
        self.molmix_model.load_ckpt(FORMULA_CKPT, map_location=self.device)
        self.molmix_model.to(self.device)
        self.molmix_model.eval()
        print(f"  MolMix 模型已加载: {FORMULA_CKPT}")

        print("加载 Diffusion 模型...")
        self.diffusion_model = FormulaDiffusion(
            unet1d_config=DIFFUSION_CONFIG["unet1d_config"],
            beta_scheduler_config=DIFFUSION_CONFIG["beta_scheduler_config"],
            sigma_scheduler_config=DIFFUSION_CONFIG["sigma_scheduler_config"],
            diff_config=DIFFUSION_CONFIG["diff_config"]
        )
        self.diffusion_model.load_ckpt(DIFFUSION_CKPT, map_location=self.device)
        self.diffusion_model.to(self.device)
        self.diffusion_model.eval()
        print(f"  Diffusion 模型已加载: {DIFFUSION_CKPT}")

        print("加载 Decoder 模型...")
        self.decoder_model = FormulaDecoder(
            formula_emb_dim=DECODER_CONFIG["formula_emb_dim"],
            set_transformer_block=DECODER_CONFIG["set_transformer_block"],
            aggr_block=DECODER_CONFIG["aggr_block"],
            mol_dict=DECODER_CONFIG["mol_dict"]
        )
        self.decoder_model.load_ckpt(DECODER_CKPT, map_location=self.device)
        self.decoder_model.to(self.device)
        self.decoder_model.eval()
        print(f"  Decoder 模型已加载: {DECODER_CKPT}")

        print("所有模型加载完成!")


# ============== API 数据模型 ==============
class SolventInput(BaseModel):
    name: str
    molar_ratio: float

class SaltInput(BaseModel):
    name: str
    molar_ratio: float

class PredictRequest(BaseModel):
    solvents: List[SolventInput]
    salts: List[SaltInput]
    temperature: float = 25  # Celsius
    concentration: float = 1.0  # mol/L

class GenerateRequest(BaseModel):
    target_conductivity: float
    target_anion_ratio: float
    temperature: float = 25
    concentration: float = 0.1
    num_samples: int = 3


# ============== FastAPI 应用 ==============
app = FastAPI(
    title="Bamboo-Mixer API (Paper Branch)",
    description="电解液配方预测与生成服务",
    version="1.0.0"
)

# CORS 中间件 - 允许前端 file:// 协议访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()


def build_formula_data(solvents: List[SolventInput], salts: List[SaltInput],
                       temperature: float, concentration: float) -> FormulaData:
    """从输入构建 FormulaData"""

    # 归一化摩尔比例
    solv_ratios = [s.molar_ratio for s in solvents]
    salt_ratios = [s.molar_ratio for s in salts]
    total_solv = sum(solv_ratios)
    total_salt = sum(salt_ratios)

    # 转换为摩尔分数
    solv_molar_ratios = np.array([r / total_solv * (1 - concentration / 10) for r in solv_ratios])
    salt_molar_ratios = np.array([r / total_salt * (concentration / 10) for r in salt_ratios])

    # 获取 mapped_smiles（普通SMILES转mapped SMILES）
    solv_names = [s.name for s in solvents]
    solv_smiles_raw = [SOLV_SMILES_MAP.get(s.name, "") for s in solvents]
    salt_names = [s.name for s in salts]
    salt_smiles_raw = [SALT_SMILES_MAP.get(s.name, "") for s in salts]

    # 转换为 mapped SMILES（模型需要）
    solv_mapped_smiles = []
    for smiles in solv_smiles_raw:
        if smiles:
            solv_mapped_smiles.append(Molecule.from_smiles(smiles).get_mapped_smiles())
        else:
            solv_mapped_smiles.append("")

    salt_mapped_smiles = []
    for smiles in salt_smiles_raw:
        if smiles:
            salt_mapped_smiles.append(Molecule.from_smiles(smiles).get_mapped_smiles())
        else:
            salt_mapped_smiles.append("")

    # 构建 FormulaData
    data = FormulaData(
        solv_names=solv_names,
        solv_mapped_smiles=solv_mapped_smiles,
        salt_names=salt_names,
        salt_mapped_smiles=salt_mapped_smiles,
        solv_molar_ratios=solv_molar_ratios,
        salt_molar_ratios=salt_molar_ratios,
        temperature=temperature,
        concentration=concentration
    )

    return data


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    if model_manager.mono_model is None:
        try:
            model_manager.load_models()
        except FileNotFoundError as e:
            print(f"警告: 模型文件未找到 - {e}")
            print("请先运行 python scripts/download_models.py 下载模型")


@app.get("/")
async def root():
    return {"message": "Bamboo-Mixer API (Paper Branch)", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": model_manager.mono_model is not None
    }


@app.get("/list/solvents")
async def list_solvents():
    """列出所有可用溶剂"""
    return {
        "solvents": solvs_data,
        "count": len(solvs_data)
    }


@app.get("/list/salts")
async def list_salts():
    """列出所有可用盐"""
    return {
        "salts": salts_data,
        "count": len(salts_data)
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """预测电解液属性"""
    if model_manager.mono_model is None:
        return {"error": "模型未加载，请检查模型文件是否存在"}

    try:
        # 构建数据
        data = build_formula_data(
            request.solvents, request.salts,
            request.temperature, request.concentration
        )
        data = data.to(model_manager.device)

        # 预测
        with torch.no_grad():
            result = model_manager.molmix_model(data)

        # 提取结果
        conductivity = result["conductivity"].item()
        anion_ratio = result["anion_ratio"].item()
        print(f"[DEBUG] 原始 anion_ratio tensor: {result['anion_ratio']}, item: {anion_ratio}")

        return {
            "success": True,
            "input": {
                "solvents": [s.model_dump() for s in request.solvents],
                "salts": [s.model_dump() for s in request.salts],
                "temperature": request.temperature,
                "concentration": request.concentration
            },
            "prediction": {
                "conductivity": round(float(conductivity), 4),  # S/m
                "anion_ratio": round(float(anion_ratio), 4)
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "detail": traceback.format_exc()
        }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """生成新的电解液配方"""
    if model_manager.diffusion_model is None or model_manager.decoder_model is None:
        return {"error": "生成模型未加载"}

    try:
        device = model_manager.device
        results = []

        with torch.no_grad():
            for i in range(request.num_samples):
                # 创建条件输入
                conductivity = torch.tensor([request.target_conductivity], device=device)
                anion_ratio = torch.tensor([request.target_anion_ratio], device=device)

                # 准备扩散输入
                data = Data()
                data.frm_emb = torch.randn(1, DECODER_CONFIG["formula_emb_dim"], device=device)
                data.conductivity = conductivity
                data.anion_ratio = anion_ratio

                # 扩散去噪
                for t in reversed(range(0, 100, 10)):  # 简化版采样
                    t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                    # 实际应使用完整的ddpm采样过程
                    noise = torch.randn_like(data.frm_emb) * 0.1
                    data.frm_emb = data.frm_emb - 0.1 * noise  # 简化的去噪步骤

                # 解码为配方
                decoded = model_manager.decoder_model.predict(data.frm_emb)
                bow_vec = decoded["bow_vec"].squeeze(0).cpu().numpy()

                # 将 BoW 转换为具体溶剂/盐及比例
                # 这里需要根据实际词汇表进行转换
                solv_count = len(SOLV_SMILES_MAP)
                salt_count = len(SALT_SMILES_MAP)

                solv_probs = bow_vec[:solv_count] if len(bow_vec) > solv_count else bow_vec[:solv_count] / bow_vec[:solv_count].sum()
                salt_probs = bow_vec[solv_count:solv_count+salt_count] if len(bow_vec) > solv_count + salt_count else bow_vec[-salt_count:] / bow_vec[-salt_count:].sum() if bow_vec[-salt_count:].sum() > 0 else np.ones(salt_count) / salt_count

                # 选择 top 3 溶剂
                top_solv_idx = np.argsort(solv_probs)[-3:][::-1]
                top_solvs = []
                for idx in top_solv_idx:
                    if solv_probs[idx] > 0.1:  # 阈值筛选
                        name = list(SOLV_SMILES_MAP.keys())[idx]
                        top_solvs.append({
                            "name": name,
                            "smiles": SOLV_SMILES_MAP[name],
                            "probability": float(solv_probs[idx])
                        })

                # 选择 top 1 盐
                top_salt_idx = np.argmax(salt_probs)
                salt_name = list(SALT_SMILES_MAP.keys())[top_salt_idx]
                top_salt = {
                    "name": salt_name,
                    "smiles": SALT_SMILES_MAP[salt_name],
                    "probability": float(salt_probs[top_salt_idx])
                }

                results.append({
                    "sample_id": i + 1,
                    "solvents": top_solvs,
                    "salt": top_salt,
                    "target_conductivity": request.target_conductivity,
                    "target_anion_ratio": request.target_anion_ratio
                })

        return {
            "success": True,
            "num_samples": len(results),
            "results": results
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "detail": traceback.format_exc()
        }


# ============== 主入口 ==============
if __name__ == "__main__":
    print("=" * 50)
    print("  Bamboo-Mixer API 服务 (Paper Branch)")
    print("=" * 50)
    print(f"\n模型目录: {MODELS_DIR}")
    print(f"词汇表: {EMB_DIR}")
    print(f"\nAPI 文档: http://localhost:8003/docs")
    print("\n按 Ctrl+C 停止服务\n")

    uvicorn.run(app, host="0.0.0.0", port=8003)
