# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

BOARD_SIZE = 15
BOARD_POSITIONS = BOARD_SIZE * BOARD_SIZE

class AttentionFusion(nn.Module):
    """注意力融合模块 - 为每个头动态分配权重"""
    def __init__(self, feat_dim=32, d_model=64, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.total_dim = feat_dim * 3  # 96
        
        # 为每个头生成注意力权重
        self.fear_attn = nn.Linear(feat_dim, 1)
        self.greed_attn = nn.Linear(feat_dim, 1)
        self.normal_attn = nn.Linear(feat_dim, 1)
        
        # 融合后的处理
        self.fusion = nn.Sequential(
            nn.Linear(self.total_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, fear, greed, normal):
        # fear, greed, normal: [batch, 225, 32]
        
        # 每个头自己决定自己的重要性
        fear_weight = torch.sigmoid(self.fear_attn(fear))  # [batch, 225, 1]
        greed_weight = torch.sigmoid(self.greed_attn(greed))
        normal_weight = torch.sigmoid(self.normal_attn(normal))
        
        # 加权特征
        weighted_fear = fear * fear_weight
        weighted_greed = greed * greed_weight
        weighted_normal = normal * normal_weight
        
        # 拼接
        combined = torch.cat([weighted_fear, weighted_greed, weighted_normal], dim=-1)  # [batch, 225, 96]
        
        # 融合
        return self.fusion(combined)  # [batch, 225, 1]

class FearGreedWuziqiModel(nn.Module):
    """五子棋恐惧与贪婪模型 - 注意力融合版"""
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.board_size = BOARD_SIZE
        self.num_positions = BOARD_POSITIONS
        
        # 基础嵌入
        self.state_embedding = nn.Embedding(3, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, BOARD_POSITIONS, d_model) * 0.02)
        self.turn_embedding = nn.Embedding(2, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
        
        # 恐惧头 - 每个位置输出32维特征
        self.fear_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),  # 32维
            nn.ReLU(),
        )
        
        # 贪婪头 - 每个位置输出32维特征
        self.greed_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),  # 32维
            nn.ReLU(),
        )
        
        # 普通策略头 - 每个位置输出32维特征
        self.normal_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),  # 32维
            nn.ReLU(),
        )
        
        # 注意力融合层
        self.position_fusion = AttentionFusion(feat_dim=32, d_model=64, dropout=dropout)
        
        # 价值头（用全局特征）
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )
        
        # 注意力头（用于可视化）
        self.attention_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, board, turn=None):
        """标准前向传播"""
        return self.forward_with_mask(board, turn)
    
    def forward_with_mask(self, board, turn=None, legal_moves=None, return_details=False):
        """
        带合法移动掩码的前向传播 - 注意力融合版
        legal_moves: list of lists, 每个样本的合法位置列表
        """
        batch_size = board.size(0)
        
        # 基础嵌入
        emb = self.state_embedding(board) * math.sqrt(self.d_model)
        emb = emb + self.pos_embedding
        
        # 加入回合信息
        if turn is not None:
            turn_emb = self.turn_embedding(turn).unsqueeze(1).expand(-1, self.num_positions, -1)
            emb = emb + turn_emb
        
        # Transformer编码器
        memory = self.encoder(emb)
        memory = self.norm(memory)
        
        # 全局特征（用于价值评估）
        global_feat = memory.mean(dim=1)
        
        # 三个头 - 每个位置输出32维特征 [batch, 225, 32]
        fear_features = self.fear_head(memory)
        greed_features = self.greed_head(memory)
        normal_features = self.normal_head(memory)
        
        # 注意力融合 - 对每个位置独立决策
        position_scores = self.position_fusion(fear_features, greed_features, normal_features)  # [batch, 225, 1]
        
        # 去掉最后一维，得到 [batch, 225]
        policy_logits = position_scores.squeeze(-1)
        
        # 如果有legal_moves，只计算合法位置的概率
        if legal_moves is not None:
            mask = torch.ones_like(policy_logits) * float('-inf')
            for b in range(batch_size):
                for pos in legal_moves[b]:
                    if 0 <= pos < self.num_positions:
                        mask[b, pos] = 0
            policy_logits = policy_logits + mask
        
        # 价值评估
        value = self.value_head(global_feat)
        
        if return_details:
            # 为了可视化，对特征取平均得到每个位置的分数（并压缩到0-1）
            # fear_scores = torch.sigmoid(fear_features.mean(dim=-1))      # [batch, 225]
            # greed_scores = torch.sigmoid(greed_features.mean(dim=-1))    # [batch, 225]
            # normal_scores = torch.sigmoid(normal_features.mean(dim=-1))  # [batch, 225]
            fear_scores = fear_features.mean(dim=-1)      # [batch, 225]
            greed_scores = greed_features.mean(dim=-1)    # [batch, 225]
            normal_scores = normal_features.mean(dim=-1)  # [batch, 225]
            
            return {
                'policy': policy_logits,
                'value': value,
                'fear_scores': fear_scores,
                'greed_scores': greed_scores,
                'normal_scores': normal_scores,
                'attention': self.attention_head(global_feat)
            }
        
        return policy_logits, value
    
    def decide_move_fast(self, board, player, device='cpu', debug=False):
        """快速决策 - 只考虑附近位置"""
        self.eval()
        with torch.no_grad():
            from game import get_nearby_moves
            
            nearby = get_nearby_moves(board, distance=2)
            if not nearby:
                center = BOARD_SIZE // 2
                center_pos = center * BOARD_SIZE + center
                return {
                    'move': center_pos,
                    'policy': np.zeros(BOARD_POSITIONS),
                    'fear': np.zeros(BOARD_POSITIONS),
                    'greed': np.zeros(BOARD_POSITIONS),
                    'normal': np.zeros(BOARD_POSITIONS),
                    'value': 0.0,
                    'attention': 0.5,
                    'nearby': [center_pos]
                }
            
            board_tensor = torch.tensor(board, dtype=torch.long, device=device).unsqueeze(0)
            turn = torch.tensor([0 if player == 1 else 1], device=device)
            
            details = self.forward_with_mask(
                board_tensor, turn, legal_moves=[nearby], return_details=True
            )
            
            policy = F.softmax(details['policy'][0], dim=-1).cpu().numpy()
            fear_scores = details['fear_scores'][0].cpu().numpy()
            greed_scores = details['greed_scores'][0].cpu().numpy()
            normal_scores = details['normal_scores'][0].cpu().numpy()
            
            nearby_probs = {pos: policy[pos] for pos in nearby}
            best_move = max(nearby_probs.items(), key=lambda x: x[1])[0]
            
            value = details['value'][0].item()
            attention = details['attention'][0].item()
            
            if debug:
                print(f"\n🎯 候选位置 ({len(nearby)}个):")
                sorted_moves = sorted(nearby_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                for pos, prob in sorted_moves:
                    from game import pos_to_str
                    print(f"   {pos_to_str(pos)}: 概率={prob:.3f}, 恐惧={fear_scores[pos]:.3f}, 贪婪={greed_scores[pos]:.3f}, 普通={normal_scores[pos]:.3f}")
            
            return {
                'move': best_move,
                'policy': policy,
                'fear': fear_scores,
                'greed': greed_scores,
                'normal': normal_scores,
                'value': value,
                'attention': attention,
                'nearby': nearby
            }
