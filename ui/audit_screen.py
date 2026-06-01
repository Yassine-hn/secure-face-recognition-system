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
		self._page_index = 0
		self._page_size = 10
		self._has_next_page = False
		self.status_var = ctk.StringVar(value="Audit log not loaded yet.")
		self.page_var = ctk.StringVar(value="Page 1")
		self._rows: list[dict[str, Any]] = []
		self.prev_button: ctk.CTkButton | None = None
		self.next_button: ctk.CTkButton | None = None

		self._build_body()
		self.refresh_entries()

	def _build_body(self) -> None:
		body = ctk.CTkFrame(self)
		body.pack(fill="both", expand=True, padx=20, pady=(0, 20))

		toolbar = ctk.CTkFrame(body)
		toolbar.pack(fill="x", padx=20, pady=(20, 10))
		ctk.CTkButton(toolbar, text="Refresh", command=self.refresh_entries).pack(side="left", padx=10, pady=10)
		self.prev_button = ctk.CTkButton(toolbar, text="Previous", command=self.show_previous_page)
		self.prev_button.pack(side="left", padx=10, pady=10)
		self.next_button = ctk.CTkButton(toolbar, text="Next", command=self.show_next_page)
		self.next_button.pack(side="left", padx=10, pady=10)
		ctk.CTkButton(toolbar, text="Export CSV", command=self.export_csv).pack(side="left", padx=10, pady=10)
		ctk.CTkLabel(toolbar, textvariable=self.page_var).pack(side="right", padx=10)

		ctk.CTkLabel(body, textvariable=self.status_var, anchor="w", justify="left").pack(fill="x", padx=32, pady=(0, 10))

		self.listing = ctk.CTkScrollableFrame(body)
		self.listing.pack(fill="both", expand=True, padx=20, pady=(0, 20))

	def refresh_entries(self) -> None:
		self._load_page(self._page_index)

	def show_previous_page(self) -> None:
		if self._page_index <= 0:
			return
		self._load_page(self._page_index - 1)

	def show_next_page(self) -> None:
		if not self._has_next_page:
			return
		self._load_page(self._page_index + 1)

	def on_show(self) -> None:
		self.refresh_entries()

	def _load_page(self, page_index: int) -> None:
		for child in self.listing.winfo_children():
			child.destroy()
		try:
			rows, has_next_page = self.services.audit_logger.read_entries_page(
				page_size=self._page_size,
				page_index=page_index,
			)
			self._page_index = page_index
			self._rows = rows
			self._has_next_page = has_next_page
			self.page_var.set(f"Page {self._page_index + 1}")
			if self.prev_button is not None:
				self.prev_button.configure(state="normal" if self._page_index > 0 else "disabled")
			if self.next_button is not None:
				self.next_button.configure(state="normal" if self._has_next_page else "disabled")
			if not self._rows:
				self.status_var.set("No audit entries found." if self._page_index == 0 else "No older audit entries found.")
				ctk.CTkLabel(self.listing, text="No entries on this page.").pack(anchor="w", padx=12, pady=12)
				return
			self.status_var.set(f"Loaded {len(self._rows)} entries on page {self._page_index + 1}.")
			for row in reversed(self._rows):
				timestamp = row.get("timestamp") or "-"
				activity_type = row.get("activity_type") or "-"
				result = row.get("result") or "-"
				student_id = row.get("student_id") or "anonymous"
				session_token = row.get("session_token")
				token_preview = f"{str(session_token)[:12]}..." if session_token else "-"
				text = f"{timestamp} | {activity_type} | {result} | {student_id} | {token_preview}"
				ctk.CTkLabel(self.listing, text=text, anchor="w", justify="left").pack(fill="x", padx=12, pady=6)
		except Exception as exc:
			self.status_var.set(f"Could not load audit log: {exc}")

	def export_csv(self) -> None:
		try:
			export_path = Path(self.services.audit_logger.log_path).with_suffix(".csv")
			with export_path.open("w", newline="", encoding="utf-8") as file:
				writer = csv.DictWriter(file, fieldnames=["timestamp", "activity_type", "result", "student_id", "student_hash", "session_token", "metadata", "prev_hmac", "hmac"])
				writer.writeheader()
				for row in self.services.audit_logger.read_entries():
					writer.writerow(row)
			self.status_var.set(f"Exported CSV to {export_path.name}")
		except Exception as exc:
			self.status_var.set(f"Export failed: {exc}")
