import json
from pathlib import Path
import tempfile
import unittest

from security.audit_log import (
	AuditLogIntegrityError,
	AuditLogger,
	append_event,
	hash_student_id,
	read_entries,
	read_entries_page,
	verify_audit_chain,
)


class TestAuditLog(unittest.TestCase):
	def setUp(self) -> None:
		self.temp_dir = tempfile.TemporaryDirectory()
		self.log_path = Path(self.temp_dir.name) / "audit.log"
		self.secret_key = "super-secret-test-key"

	def tearDown(self) -> None:
		self.temp_dir.cleanup()

	def test_append_and_verify_chain(self) -> None:
		logger = AuditLogger(log_path=self.log_path, secret_key=self.secret_key)
		first = logger.append_event(
			result="DENIED",
			student_id="S001",
			session_token="token-1",
		)
		second = logger.append_event(
			result="GRANTED",
			student_id="S001",
			session_token="token-2",
		)

		self.assertEqual(second["prev_hmac"], first["hmac"])
		self.assertTrue(logger.verify())
		self.assertTrue(verify_audit_chain(self.log_path, self.secret_key))

	def test_tampering_is_detected(self) -> None:
		logger = AuditLogger(log_path=self.log_path, secret_key=self.secret_key)
		logger.append_event(result="DENIED", student_id="S001", session_token="token-1")
		logger.append_event(result="GRANTED", student_id="S001", session_token="token-2")

		lines = self.log_path.read_text(encoding="utf-8").splitlines()
		first_entry = json.loads(lines[0])
		first_entry["result"] = "GRANTED"
		lines[0] = json.dumps(first_entry, sort_keys=True)
		self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

		with self.assertRaises(AuditLogIntegrityError):
			verify_audit_chain(self.log_path, self.secret_key)

	def test_hash_student_id_is_deterministic(self) -> None:
		h1 = hash_student_id("S001")
		h2 = hash_student_id("S001")
		self.assertEqual(h1, h2)
		self.assertNotEqual(h1, "S001")
		self.assertEqual(len(h1), 64)

	def test_functional_append_and_read(self) -> None:
		append_event(
			log_path=self.log_path,
			secret_key=self.secret_key,
			result="DENIED",
			student_id="S002",
			session_token="token-x",
			metadata={"reason": "liveness_failed"},
		)
		entries = list(read_entries(self.log_path))
		self.assertEqual(len(entries), 1)
		self.assertEqual(entries[0]["result"], "DENIED")
		self.assertEqual(entries[0]["metadata"]["reason"], "liveness_failed")

	def test_paged_read_returns_ten_entries_and_next_flag(self) -> None:
		logger = AuditLogger(log_path=self.log_path, secret_key=self.secret_key)
		for index in range(1, 26):
			logger.append_event(
				result="GRANTED" if index % 2 == 0 else "DENIED",
				student_id=f"S{index:03d}",
				session_token=f"token-{index}",
			)

		page1, has_next1 = read_entries_page(self.log_path, page_size=10, page_index=0)
		page2, has_next2 = read_entries_page(self.log_path, page_size=10, page_index=1)
		page3, has_next3 = read_entries_page(self.log_path, page_size=10, page_index=2)

		self.assertEqual(len(page1), 10)
		self.assertEqual(len(page2), 10)
		self.assertEqual(len(page3), 5)
		self.assertTrue(has_next1)
		self.assertTrue(has_next2)
		self.assertFalse(has_next3)
		self.assertEqual([row["session_token"] for row in page1], [f"token-{index}" for index in range(16, 26)])
		self.assertEqual([row["session_token"] for row in page2], [f"token-{index}" for index in range(6, 16)])
		self.assertEqual([row["session_token"] for row in page3], [f"token-{index}" for index in range(1, 6)])


if __name__ == "__main__":
	unittest.main()
