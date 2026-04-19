"""Settings screen for tuning similarity and liveness thresholds."""

from __future__ import annotations

from typing import Any

import customtkinter as ctk

from ui.base import ScreenFrame


class SettingsScreen(ScreenFrame):
	def __init__(self, master: Any, services: Any, **kwargs: Any) -> None:
		super().__init__(
			master,
			title="Settings",
			subtitle="Adjust recognition and liveness thresholds for your current session.",
			**kwargs,
		)
		self.services = services
		self.matcher_threshold = ctk.DoubleVar(value=float(self.services.matcher.threshold))
		self.passive_threshold = ctk.DoubleVar(value=float(self.services.passive_liveness.threshold))
		self.window_seconds = ctk.DoubleVar(value=float(self.services.active_liveness.window_seconds))
		self.status_var = ctk.StringVar(value="Adjust values and save them to the live services.")

		self._build_body()

	def _build_body(self) -> None:
		panel = ctk.CTkFrame(self)
		panel.pack(fill="both", expand=True, padx=20, pady=(0, 20))

		content = ctk.CTkFrame(panel)
		content.pack(fill="both", expand=True, padx=20, pady=20)

		self._add_slider(content, "Matcher threshold", self.matcher_threshold, 0.1, 0.95, self._apply)
		self._add_slider(content, "Passive liveness threshold", self.passive_threshold, 0.1, 0.95, self._apply)
		self._add_slider(content, "Active window (seconds)", self.window_seconds, 1.0, 5.0, self._apply)

		ctk.CTkLabel(content, textvariable=self.status_var).pack(anchor="w", padx=20, pady=(20, 10))
		ctk.CTkButton(content, text="Apply Settings", command=self._apply).pack(anchor="w", padx=20, pady=(0, 20))

	def _add_slider(self, master: Any, label: str, variable: ctk.Variable, start: float, end: float, command) -> None:
		ctk.CTkLabel(master, text=label, font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=(20, 6))
		slider = ctk.CTkSlider(master, from_=start, to=end, number_of_steps=20, variable=variable, command=lambda _value: command())
		slider.pack(fill="x", padx=20)
		ctk.CTkLabel(master, textvariable=variable).pack(anchor="w", padx=20, pady=(4, 0))

	def _apply(self) -> None:
		try:
			self.services.matcher.set_threshold(float(self.matcher_threshold.get()))
			self.services.passive_liveness.threshold = float(self.passive_threshold.get())
			self.services.active_liveness.window_seconds = float(self.window_seconds.get())
			self.status_var.set("Settings applied to live services.")
		except Exception as exc:
			self.status_var.set(f"Could not apply settings: {exc}")
