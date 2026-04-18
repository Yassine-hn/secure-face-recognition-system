"""Audit screen for viewing tamper-evident access events."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import customtkinter as ctk

from ui.base import ScreenFrame


class AuditScreen(ScreenFrame):
	def __init__(self, master: Any, services: Any, **kwargs: Any) -> None:
		super().__init__(
			master,
			title="Audit Log",
			subtitle="Review access events and export the hash-chained log for offline inspection.",
			**kwargs,
		)
		self.services = services
		self.status_var = ctk.StringVar(value="Audit log not loaded yet.")
		self._rows: list[dict[str, Any]] = []

		self._build_body()
		self.refresh_entries()

	def _build_body(self) -> None:
		body = ctk.CTkFrame(self)
		body.pack(fill="both", expand=True, padx=20, pady=(0, 20))

		toolbar = ctk.CTkFrame(body)
		toolbar.pack(fill="x", padx=20, pady=(20, 10))
		ctk.CTkButton(toolbar, text="Refresh", command=self.refresh_entries).pack(side="left", padx=10, pady=10)
		ctk.CTkButton(toolbar, text="Export CSV", command=self.export_csv).pack(side="left", padx=10, pady=10)
		ctk.CTkLabel(toolbar, textvariable=self.status_var).pack(side="right", padx=10)

		self.listing = ctk.CTkScrollableFrame(body)
		self.listing.pack(fill="both", expand=True, padx=20, pady=(0, 20))

	def refresh_entries(self) -> None:
		for child in self.listing.winfo_children():
			child.destroy()
		try:
			self._rows = list(self.services.audit_logger.read_entries())
			if not self._rows:
				self.status_var.set("No audit entries found.")
				ctk.CTkLabel(self.listing, text="No entries yet.").pack(anchor="w", padx=12, pady=12)
				return
			self.status_var.set(f"Loaded {len(self._rows)} entries.")
			for row in reversed(self._rows):
				text = f"{row.get('timestamp')} | {row.get('result')} | {row.get('student_hash', 'anonymous')} | {row.get('session_token', '-')[:12]}..."
				ctk.CTkLabel(self.listing, text=text, anchor="w", justify="left").pack(fill="x", padx=12, pady=6)
		except Exception as exc:
			self.status_var.set(f"Could not load audit log: {exc}")

	def export_csv(self) -> None:
		try:
			export_path = Path(self.services.audit_logger.log_path).with_suffix(".csv")
			with export_path.open("w", newline="", encoding="utf-8") as file:
				writer = csv.DictWriter(file, fieldnames=["timestamp", "result", "student_hash", "session_token", "metadata", "prev_hmac", "hmac"])
				writer.writeheader()
				for row in self._rows:
					writer.writerow(row)
			self.status_var.set(f"Exported CSV to {export_path.name}")
		except Exception as exc:
			self.status_var.set(f"Export failed: {exc}")
