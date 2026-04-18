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


if __name__ == "__main__":
	unittest.main()
