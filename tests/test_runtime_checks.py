import os
import unittest
from unittest import mock

from security.runtime_checks import (
    allow_plaintext_database_fallback,
    require_keyring_only,
    validate_runtime_security,
)


class TestRuntimeChecks(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            "SFRS_MODE": os.getenv("SFRS_MODE"),
            "SFRS_KEYRING_ONLY": os.getenv("SFRS_KEYRING_ONLY"),
            "SFRS_ALLOW_PLAINTEXT_DB": os.getenv("SFRS_ALLOW_PLAINTEXT_DB"),
            "DB_KEY": os.getenv("DB_KEY"),
        }

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_require_keyring_only_in_production(self) -> None:
        os.environ["SFRS_MODE"] = "production"
        self.assertTrue(require_keyring_only())

    def test_allow_plaintext_disabled_in_production(self) -> None:
        os.environ["SFRS_MODE"] = "production"
        self.assertFalse(allow_plaintext_database_fallback())

    def test_validate_security_rejects_env_db_key_in_keyring_only_mode(self) -> None:
        os.environ["SFRS_MODE"] = "production"
        os.environ["DB_KEY"] = "test-secret"
        ok, issues = validate_runtime_security()
        self.assertFalse(ok)
        self.assertTrue(any("DB_KEY" in issue for issue in issues))

    @mock.patch("security.runtime_checks._module_available")
    def test_validate_security_rejects_missing_sqlcipher_when_plaintext_forbidden(self, module_available) -> None:
        os.environ["SFRS_MODE"] = "production"

        def fake_available(name: str) -> bool:
            return name != "sqlcipher3"

        module_available.side_effect = fake_available
        ok, issues = validate_runtime_security()
        self.assertFalse(ok)
        self.assertTrue(any("sqlcipher3" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
