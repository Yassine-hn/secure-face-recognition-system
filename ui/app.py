"""Main CustomTkinter application window and screen router."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import customtkinter as ctk

from core.services import AppServices, build_default_services


class SecureFaceRecognitionApp(ctk.CTk):
	"""Root window that owns and switches between screens."""

	def __init__(self, services: AppServices | None = None) -> None:
		super().__init__()
		self.title("Secure Face Recognition System")
		self.geometry("1280x780")
		self.minsize(1100, 700)
		ctk.set_appearance_mode("dark")
		ctk.set_default_color_theme("blue")

		self.services = services or build_default_services()
		self._screens: dict[str, ctk.CTkFrame] = {}
		self._active_screen: ctk.CTkFrame | None = None

		self.sidebar = ctk.CTkFrame(self, width=220)
		self.sidebar.pack(side="left", fill="y")

		self.content = ctk.CTkFrame(self)
		self.content.pack(side="right", fill="both", expand=True)

		self._build_sidebar()
		self.show_screen("recognition")

	def _build_sidebar(self) -> None:
		title = ctk.CTkLabel(self.sidebar, text="Secure Face Recognition", font=ctk.CTkFont(size=20, weight="bold"))
		title.pack(padx=16, pady=(18, 10))

		nav_items = [
			("Recognition", "recognition"),
			("Enrollment", "enrollment"),
			("Liveness", "liveness"),
			("Audit", "audit"),
			("Settings", "settings"),
		]
		for label, screen_name in nav_items:
			button = ctk.CTkButton(self.sidebar, text=label, command=lambda name=screen_name: self.show_screen(name))
			button.pack(fill="x", padx=16, pady=6)

	def _create_screen(self, name: str) -> ctk.CTkFrame:
		from ui.audit_screen import AuditScreen
		from ui.enrollment_screen import EnrollmentScreen
		from ui.liveness_screen import LivenessScreen
		from ui.recognition_screen import RecognitionScreen
		from ui.settings_screen import SettingsScreen

		factories: dict[str, Callable[[Any], ctk.CTkFrame]] = {
			"recognition": lambda master: RecognitionScreen(master, services=self.services),
			"enrollment": lambda master: EnrollmentScreen(master, services=self.services),
			"liveness": lambda master: LivenessScreen(master, services=self.services),
			"audit": lambda master: AuditScreen(master, services=self.services),
			"settings": lambda master: SettingsScreen(master, services=self.services),
		}
		factory = factories.get(name)
		if factory is None:
			raise KeyError(f"Unknown screen {name!r}")
		return factory(self.content)

	def show_screen(self, name: str) -> None:
		if self._active_screen is not None and hasattr(self._active_screen, "on_hide"):
			try:
				self._active_screen.on_hide()
			except Exception:
				pass
		for child in self.content.winfo_children():
			child.pack_forget()
		screen = self._screens.get(name)
		if screen is None:
			screen = self._create_screen(name)
			self._screens[name] = screen
		screen.pack(fill="both", expand=True)
		self._active_screen = screen
		if hasattr(screen, "on_show"):
			try:
				screen.on_show()
			except Exception:
				pass


def run_app() -> None:
	app = SecureFaceRecognitionApp()
	app.mainloop()

