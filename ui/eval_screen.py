"""Evaluation screen for offline pipeline review."""

from __future__ import annotations

from typing import Any

import customtkinter as ctk

from ui.base import ScreenFrame


class EvaluationScreen(ScreenFrame):
	def __init__(self, master: Any, services: Any, **kwargs: Any) -> None:
		super().__init__(
			master,
			title="Evaluation",
			subtitle="Review the offline evaluation entry points and run lightweight status checks.",
			**kwargs,
		)
		self.services = services
		self.status_var = ctk.StringVar(value="Ready for evaluation tasks.")

		self._build_body()

	def _build_body(self) -> None:
		panel = ctk.CTkFrame(self)
		panel.pack(fill="both", expand=True, padx=20, pady=(0, 20))

		ctk.CTkLabel(
			panel,
			text="Offline evaluation is handled by the `evaluation/` package.",
			wraplength=900,
			justify="left",
		).pack(anchor="w", padx=20, pady=(24, 10))
		ctk.CTkLabel(
			panel,
			text="Use the benchmark and attack simulation modules from the command line for full ROC/EER analysis.",
			wraplength=900,
			justify="left",
		).pack(anchor="w", padx=20, pady=(0, 20))
		ctk.CTkLabel(panel, textvariable=self.status_var).pack(anchor="w", padx=20, pady=(0, 20))
		ctk.CTkButton(panel, text="Refresh Status", command=self.refresh_status).pack(anchor="w", padx=20)

	def refresh_status(self) -> None:
		self.status_var.set("UI evaluation panel is connected; run offline metrics from the CLI for now.")
