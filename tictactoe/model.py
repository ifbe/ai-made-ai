# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class FearGreedModel(nn.Module):
    """恐惧与贪婪模型 - 可学习的优先级融合"""
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 基础嵌入
        self.state_embedding = nn.Embedding(3, d_model)  # 0=空, 1=X, 2=O
        self.pos_embedding = nn.Parameter(torch.randn(1, 9, d_model) * 0.02)
        self.turn_embedding = nn.Embedding(2, d_model)  # 0=X先手, 1=O后手
        
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
        
        # 恐惧头 - 输出特征
        self.fear_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 贪婪头 - 输出特征
        self.greed_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 普通策略头 - 输出特征
        self.normal_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 优先级融合层 - 直接输出最终策略
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 3 // 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 9)
        )
        
        # 价值头 - 局面评估
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )
        
        # 注意力头 - 用于可视化
        self.attention_head = nn.Sequential(
            nn.Linear(d_model * 3 // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, board, turn=None, return_details=False):
        batch_size = board.size(0)
        
        # 嵌入
        emb = self.state_embedding(board) * math.sqrt(self.d_model)
        emb = emb + self.pos_embedding
        
        # 添加turn信息
        if turn is not None:
            turn_emb = self.turn_embedding(turn).unsqueeze(1)
            emb = torch.cat([turn_emb, emb], dim=1)
            has_turn = True
        else:
            has_turn = False
        
        # Encoder
        memory = self.encoder(emb)
        memory = self.norm(memory)
        
        # 提取位置特征
        if has_turn:
            position_features = memory[:, 1:, :]
            global_feat = memory[:, 0, :]
        else:
            position_features = memory
            global_feat = memory.mean(dim=1)
        
        # 三个专家头提取特征
        fear_features = self.fear_head(position_features)
        greed_features = self.greed_head(position_features)
        normal_features = self.normal_head(position_features)
        
        # 拼接三个头的特征
        combined_features = torch.cat([
            fear_features, 
            greed_features, 
            normal_features
        ], dim=-1)
        
        # 对每个位置取平均，得到全局特征
        global_combined = combined_features.mean(dim=1)
        
        # 优先级融合层 - 直接输出最终策略
        final_policy = self.fusion_layer(global_combined)
        
        # 价值评估
        value = self.value_head(global_feat)
        
        # 注意力分数（用于可视化）
        attention = self.attention_head(global_combined)
        
        if return_details:
            fear_scores = fear_features.mean(dim=-1)
            greed_scores = greed_features.mean(dim=-1)
            
            return {
                'policy': final_policy,
                'value': value,
                'fear_features': fear_features,
                'greed_features': greed_features,
                'normal_features': normal_features,
                'fear_scores': fear_scores,
                'greed_scores': greed_scores,
                'attention': attention,
                'global_feat': global_feat
            }
        
        return final_policy, value
    
    def decide_move(self, board, player, device='cpu', debug=False):
        self.eval()
        with torch.no_grad():
            board_tensor = torch.tensor(board, dtype=torch.long, device=device).unsqueeze(0)
            turn = torch.tensor([0 if player == 1 else 1], device=device)
            
            details = self.forward(board_tensor, turn, return_details=True)
            
            policy = F.softmax(details['policy'][0], dim=-1).cpu().numpy()
            fear_scores = details['fear_scores'][0].cpu().numpy()
            greed_scores = details['greed_scores'][0].cpu().numpy()
            value = details['value'][0].item()
            attention = details['attention'][0].item()
            
            from game import get_legal_moves
            legals = get_legal_moves(board)
            
            valid_policy = policy.copy()
            for i in range(9):
                if i not in legals:
                    valid_policy[i] = 0
            if valid_policy.sum() > 0:
                valid_policy = valid_policy / valid_policy.sum()
            else:
                valid_policy[legals[0]] = 1.0
            
            best_move = valid_policy.argmax()
            
            if debug:
                print(f"\n📊 优先级融合结果:")
                print(f"   注意力值: {attention:.4f}")
                print("\n   恐惧分数:")
                for i in range(3):
                    row = ""
                    for j in range(3):
                        idx = i * 3 + j
                        row += f" {fear_scores[idx]:.2f} "
                    print(f"      {row}")
                print("\n   贪婪分数:")
                for i in range(3):
                    row = ""
                    for j in range(3):
                        idx = i * 3 + j
                        row += f" {greed_scores[idx]:.2f} "
                    print(f"      {row}")
            
            return {
                'move': best_move,
                'policy': valid_policy,
                'fear': fear_scores,
                'greed': greed_scores,
                'value': value,
                'attention': attention
            }
