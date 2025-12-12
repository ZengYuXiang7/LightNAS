class TransNAS(nn.Module):
    def __init__(self, enc_in, config):
        super(TransNAS, self).__init__()
        self.config = config
        self.d_model = config.d_model

        # --- 新增配置：融合模式 ---
        # 'concat': 拼接 (推荐, 维度变大, 表达力更强)
        # 'add': 加权求和 (维度不变, 节省参数)
        self.merge_mode = "concat"

        # 1. 基础编码器 (共享权重)
        self.op_embedding = DiscreteEncoder(
            num_operations=8,
            encoding_dim=self.d_model,
            encoding_type=config.op_encoder,
            output_dim=self.d_model,
        )
        self.indeg_embedding = DiscreteEncoder(
            num_operations=10,
            encoding_dim=self.d_model,
            encoding_type="embedding",
            output_dim=self.d_model,
        )
        self.outdeg_embedding = DiscreteEncoder(
            num_operations=10,
            encoding_dim=self.d_model,
            encoding_type="embedding",
            output_dim=self.d_model,
        )

        self.lap_encoder = nn.Linear(1 * config.lp_d_model, self.d_model, bias=True)
        self.att_bias = SPDSpatialBias(num_heads=config.num_heads, max_dist=99)

        # Graph Token (共享)
        self.graph_tok = nn.Parameter(torch.randn(1, 1, self.d_model))

        # 如果是加权求和，定义一个可学习的参数 alpha
        if self.merge_mode == "add":
            self.merge_weight = nn.Parameter(torch.tensor(0.5))

        # Transformer Encoder (共享权重)
        self.encoder = Transformer(
            self.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            norm_method="rms",
            ffn_method="ffn",
            att_method=config.att_method,
        )

        # 2. 预测头调整
        # 如果是 'concat'，输入维度变成 2 * d_model
        # 如果是 'add'，输入维度保持 d_model
        pred_input_dim = (
            self.d_model * 2 if self.merge_mode == "concat" else self.d_model
        )

        self.pred_head = nn.Sequential(nn.Linear(pred_input_dim, 1), nn.Dropout(0.10))

    def _get_encoder_output(
        self, features, eigvec, indgree, outdegree, dij, key_padding_mask
    ):
        """
        辅助函数：单次前向传播逻辑
        """
        B, _ = features.shape

        # Embedding 叠加
        seq_embeds = (
            self.op_embedding(features)
            + self.indeg_embedding(indgree)
            + self.outdeg_embedding(outdegree)
        )
        seq_embeds = seq_embeds + self.lap_encoder(eigvec)

        # 添加 Graph Token
        graph_tok = self.graph_tok.expand(B, -1, -1)
        seq_embeds = torch.cat([graph_tok, seq_embeds], dim=1)

        # Spatial Bias
        if self.config.att_bias:
            attn_mask = self.att_bias(dij)
        else:
            attn_mask = None

        # 这里的 key_padding_mask 处理需要根据你实际情况补全
        # 假设已经在外部处理好了 Graph Token 的 mask 位

        # Transformer
        out = self.encoder(seq_embeds, attn_mask)

        # 取出 Graph Token ([CLS])
        cls_out = out[:, 0, :]
        return cls_out

    def forward(
        self, graphs, features, eigvec, indgree, outdegree, dij, key_padding_mask
    ):
        # -------------------------------------------------
        # 1. 正向流 (Forward Flow)
        # -------------------------------------------------
        cls_fwd = self._get_encoder_output(
            features,
            eigvec,
            indgree,
            outdegree,  # 正常的入度、出度
            dij,  # 正常的距离矩阵
            key_padding_mask,
        )

        # -------------------------------------------------
        # 2. 反向流 (Backward Flow)
        # -------------------------------------------------
        # 关键操作：
        # a. 交换入度和出度 (Swap In/Out Degree)
        # b. 转置距离矩阵 (Transpose Distance Matrix, dim 1 & 2)
        cls_bwd = self._get_encoder_output(
            features,
            eigvec,
            outdegree,
            indgree,  # <--- 交换传参
            dij.transpose(1, 2),  # <--- 转置矩阵
            key_padding_mask,
        )

        # -------------------------------------------------
        # 3. 融合 (Aggregation)
        # -------------------------------------------------
        if self.merge_mode == "concat":
            # 拼接: [B, 2 * d_model]
            combined = torch.cat([cls_fwd, cls_bwd], dim=1)

        elif self.merge_mode == "add":
            # 加权求和: [B, d_model]
            # 使用 Sigmoid 保证权重在 0-1 之间
            w = torch.sigmoid(self.merge_weight)
            combined = w * cls_fwd + (1 - w) * cls_bwd

        else:
            # 简单相加
            combined = cls_fwd + cls_bwd

        # 4. 预测
        y = self.pred_head(combined)

        return y
