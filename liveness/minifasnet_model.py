"""Native PyTorch implementation of MiniFASNetV2 and crop utilities.

Self-contained module providing the neural network architecture and multi-scale
face cropping for the Silent-Face-Anti-Spoofing MiniFASNet model without
requiring external repository cloning or sys.path manipulation.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
	AdaptiveAvgPool2d,
	BatchNorm1d,
	BatchNorm2d,
	Conv2d,
	Linear,
	Module,
	PReLU,
	ReLU,
	Sequential,
	Sigmoid,
)


# ── MiniFASNet Building Blocks ───────────────────────────────────────────────────

class L2Norm(Module):
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return F.normalize(x)


class Flatten(Module):
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x.view(x.size(0), -1)


class Conv_block(Module):
	def __init__(
		self,
		in_c: int,
		out_c: int,
		kernel: Tuple[int, int] = (1, 1),
		stride: Tuple[int, int] = (1, 1),
		padding: Tuple[int, int] = (0, 0),
		groups: int = 1,
	) -> None:
		super().__init__()
		self.conv = Conv2d(
			in_c, out_c, kernel_size=kernel, groups=groups,
			stride=stride, padding=padding, bias=False,
		)
		self.bn = BatchNorm2d(out_c)
		self.prelu = PReLU(out_c)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.prelu(self.bn(self.conv(x)))


class Linear_block(Module):
	def __init__(
		self,
		in_c: int,
		out_c: int,
		kernel: Tuple[int, int] = (1, 1),
		stride: Tuple[int, int] = (1, 1),
		padding: Tuple[int, int] = (0, 0),
		groups: int = 1,
	) -> None:
		super().__init__()
		self.conv = Conv2d(
			in_c, out_channels=out_c, kernel_size=kernel,
			groups=groups, stride=stride, padding=padding, bias=False,
		)
		self.bn = BatchNorm2d(out_c)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.bn(self.conv(x))


class Depth_Wise(Module):
	def __init__(
		self,
		c1: Tuple[int, int],
		c2: Tuple[int, int],
		c3: Tuple[int, int],
		residual: bool = False,
		kernel: Tuple[int, int] = (3, 3),
		stride: Tuple[int, int] = (2, 2),
		padding: Tuple[int, int] = (1, 1),
		groups: int = 1,
	) -> None:
		super().__init__()
		c1_in, c1_out = c1
		c2_in, c2_out = c2
		c3_in, c3_out = c3
		self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
		self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
		self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
		self.residual = residual

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.residual:
			short_cut = x
		out = self.project(self.conv_dw(self.conv(x)))
		if self.residual:
			return short_cut + out
		return out


class Residual(Module):
	def __init__(
		self,
		c1: Sequence[Tuple[int, int]],
		c2: Sequence[Tuple[int, int]],
		c3: Sequence[Tuple[int, int]],
		num_block: int,
		groups: int,
		kernel: Tuple[int, int] = (3, 3),
		stride: Tuple[int, int] = (1, 1),
		padding: Tuple[int, int] = (1, 1),
	) -> None:
		super().__init__()
		modules = []
		for i in range(num_block):
			modules.append(
				Depth_Wise(
					c1[i], c2[i], c3[i], residual=True,
					kernel=kernel, padding=padding, stride=stride, groups=groups,
				)
			)
		self.model = Sequential(*modules)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)


class MiniFASNet(Module):
	def __init__(
		self,
		keep: Sequence[int],
		embedding_size: int,
		conv6_kernel: Tuple[int, int] = (7, 7),
		drop_p: float = 0.2,
		num_classes: int = 3,
		img_channel: int = 3,
	) -> None:
		super().__init__()
		self.embedding_size = embedding_size

		self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
		self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

		c1 = [(keep[1], keep[2])]
		c2 = [(keep[2], keep[3])]
		c3 = [(keep[3], keep[4])]
		self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

		c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
		c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
		c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
		self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

		c1 = [(keep[16], keep[17])]
		c2 = [(keep[17], keep[18])]
		c3 = [(keep[18], keep[19])]
		self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

		c1 = [
			(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]),
			(keep[28], keep[29]), (keep[31], keep[32]), (keep[34], keep[35]),
		]
		c2 = [
			(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]),
			(keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36]),
		]
		c3 = [
			(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]),
			(keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37]),
		]
		self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

		c1 = [(keep[37], keep[38])]
		c2 = [(keep[38], keep[39])]
		c3 = [(keep[39], keep[40])]
		self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

		c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
		c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
		c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
		self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

		self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
		self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
		self.conv_6_flatten = Flatten()
		self.linear = Linear(512, embedding_size, bias=False)
		self.bn = BatchNorm1d(embedding_size)
		self.drop = nn.Dropout(p=drop_p)
		self.prob = Linear(embedding_size, num_classes, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.conv1(x)
		out = self.conv2_dw(out)
		out = self.conv_23(out)
		out = self.conv_3(out)
		out = self.conv_34(out)
		out = self.conv_4(out)
		out = self.conv_45(out)
		out = self.conv_5(out)
		out = self.conv_6_sep(out)
		out = self.conv_6_dw(out)
		out = self.conv_6_flatten(out)
		if self.embedding_size != 512:
			out = self.linear(out)
		out = self.bn(out)
		out = self.drop(out)
		out = self.prob(out)
		return out


_KEEP_DICT_1_8M_V2 = [
	32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
	13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
	52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
	128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512,
]


def MiniFASNetV2(
	embedding_size: int = 128,
	conv6_kernel: Tuple[int, int] = (5, 5),
	drop_p: float = 0.2,
	num_classes: int = 3,
	img_channel: int = 3,
) -> MiniFASNet:
	"""Construct a MiniFASNetV2 network matching 2.7_80x80_MiniFASNetV2 weights."""
	return MiniFASNet(
		_KEEP_DICT_1_8M_V2,
		embedding_size=embedding_size,
		conv6_kernel=conv6_kernel,
		drop_p=drop_p,
		num_classes=num_classes,
		img_channel=img_channel,
	)


# ── Utilities for Model Setup & Multi-Scale Face Crop ────────────────────────────

def get_kernel(height: int, width: int) -> Tuple[int, int]:
	return ((height + 15) // 16, (width + 15) // 16)


def parse_model_name(model_name: str) -> Tuple[int, int, str, Optional[float]]:
	info = model_name.split("_")[0:-1]
	h_input, w_input = info[-1].split("x")
	model_type = model_name.split(".pth")[0].split("_")[-1]

	if info[0] == "org":
		scale = None
	else:
		scale = float(info[0])
	return int(h_input), int(w_input), model_type, scale


def to_tensor(img: np.ndarray) -> torch.Tensor:
	"""Convert HxWxC BGR/RGB numpy array into 1xCxHxW float32 torch Tensor."""
	if img.ndim == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	# Transpose HxWxC -> CxHxW
	tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
	return tensor


class CropImage:
	"""Crop face patch from original image scaled by model's scale factor (e.g. 2.7)."""

	@staticmethod
	def _get_new_box(src_w: int, src_h: int, bbox: Sequence[float], scale: float) -> Tuple[int, int, int, int]:
		x, y, box_w, box_h = [float(v) for v in bbox]
		scale = min((src_h - 1) / max(1.0, box_h), min((src_w - 1) / max(1.0, box_w), scale))

		new_width = box_w * scale
		new_height = box_h * scale
		center_x = box_w / 2.0 + x
		center_y = box_h / 2.0 + y

		left_top_x = center_x - new_width / 2.0
		left_top_y = center_y - new_height / 2.0
		right_bottom_x = center_x + new_width / 2.0
		right_bottom_y = center_y + new_height / 2.0

		if left_top_x < 0:
			right_bottom_x -= left_top_x
			left_top_x = 0

		if left_top_y < 0:
			right_bottom_y -= left_top_y
			left_top_y = 0

		if right_bottom_x > src_w - 1:
			left_top_x -= right_bottom_x - src_w + 1
			right_bottom_x = src_w - 1

		if right_bottom_y > src_h - 1:
			left_top_y -= right_bottom_y - src_h + 1
			right_bottom_y = src_h - 1

		return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

	def crop(
		self,
		org_img: np.ndarray,
		bbox: Sequence[float],
		scale: float,
		out_w: int,
		out_h: int,
		crop: bool = True,
	) -> np.ndarray:
		if not crop:
			return cv2.resize(org_img, (out_w, out_h))

		src_h, src_w = org_img.shape[:2]
		x1, y1, x2, y2 = self._get_new_box(src_w, src_h, bbox, scale)
		if x2 <= x1 or y2 <= y1:
			return cv2.resize(org_img, (out_w, out_h))
		patch = org_img[y1 : y2 + 1, x1 : x2 + 1]
		if patch.size == 0:
			return cv2.resize(org_img, (out_w, out_h))
		return cv2.resize(patch, (out_w, out_h))
