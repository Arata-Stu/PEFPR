#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, SPPBottleneck


class CSPDarknet(nn.Module):
    def __init__(
        self,
        in_channels=3,         
        depth=1.0,             
        width=1.0,             
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(width * 64)  # 64
        base_depth = max(round(depth * 3), 1)  # 3

        # 各ステージの次元数とストライドを保存する辞書
        self.stage_dims = {}
        self.strides = {}

        # stem (Stride: 2)
        self.stem = Focus(in_channels, base_channels, ksize=3, act=act)
        self.stage_dims["stem"] = base_channels
        self.strides["stem"] = 2

        # dark2 (Stride: 4)
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.stage_dims["dark2"] = base_channels * 2
        self.strides["dark2"] = 4

        # dark3 (Stride: 8)
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.stage_dims["dark3"] = base_channels * 4
        self.strides["dark3"] = 8

        # dark4 (Stride: 16)
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.stage_dims["dark4"] = base_channels * 8
        self.strides["dark4"] = 16

        # dark5 (Stride: 32)
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.stage_dims["dark5"] = base_channels * 16
        self.strides["dark5"] = 32

    def get_stage_dims(self, stages: tuple) -> tuple:
        """指定されたステージ（文字列）のチャネル次元数のタプルを返す"""
        return tuple(self.stage_dims[s] for s in stages)

    def get_strides(self, stages: tuple) -> tuple:
        """指定されたステージ（文字列）の累積ストライドのタプルを返す"""
        return tuple(self.strides[s] for s in stages)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}