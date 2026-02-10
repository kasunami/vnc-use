"""Tests for PostgreSQL credential store."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from vnc_use.credential_store import VNCCredentials

# Mock psycopg2 before importing credential_store_postgres (which requires it)
mock_psycopg2 = MagicMock()
mock_psycopg2.extras = MagicMock()
mock_psycopg2.extras.RealDictCursor = MagicMock()
sys.modules["psycopg2"] = mock_psycopg2
sys.modules["psycopg2.extras"] = mock_psycopg2.extras

# Use importlib to load after mocking - avoids E402 since this is a function call
_pg_module = importlib.import_module("vnc_use.credential_store_postgres")
PostgreSQLCredentialStore = _pg_module.PostgreSQLCredentialStore
get_maistack_store = _pg_module.get_maistack_store


class TestPostgreSQLCredentialStoreInit:
    """Tests for PostgreSQLCredentialStore initialization."""

    def test_init_with_explicit_params(self):
        """Test initialization with explicit parameters."""
        store = PostgreSQLCredentialStore(
            database_url="postgresql://user:pass@localhost/db",
            encryption_key="test-key",
        )
        assert store.database_url == "postgresql://user:pass@localhost/db"
        assert store.encryption_key == "test-key"

    def test_init_from_env_vars(self, monkeypatch):
        """Test initialization from environment variables."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://env@localhost/envdb")
        monkeypatch.setenv("VNC_ENCRYPTION_KEY", "env-key")

        store = PostgreSQLCredentialStore()
        assert store.database_url == "postgresql://env@localhost/envdb"
        assert store.encryption_key == "env-key"

    def test_init_missing_database_url_raises(self, monkeypatch):
        """Test that missing DATABASE_URL raises ValueError."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with pytest.raises(ValueError, match="DATABASE_URL environment variable not set"):
            PostgreSQLCredentialStore()

    def test_init_default_encryption_key_warning(self, monkeypatch, caplog):
        """Test that default encryption key logs a warning."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
        monkeypatch.delenv("VNC_ENCRYPTION_KEY", raising=False)

        store = PostgreSQLCredentialStore()
        assert store.encryption_key == "default-key-change-me"
        assert "Using default encryption key" in caplog.text


class TestEncryption:
    """Tests for password encryption/decryption."""

    @pytest.fixture
    def store(self, monkeypatch):
        """Create a store for testing."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
        return PostgreSQLCredentialStore(encryption_key="test-encryption-key")

    def test_encrypt_decrypt_roundtrip(self, store):
        """Test that encryption and decryption are reversible."""
        password = "my-secret-password"
        encrypted = store._encrypt_password(password)
        decrypted = store._decrypt_password(encrypted)
        assert decrypted == password

    def test_encrypt_empty_password(self, store):
        """Test encrypting empty password returns empty string."""
        assert store._encrypt_password("") == ""

    def test_encrypt_without_key_raises(self, monkeypatch):
        """Test that encryption without key raises ValueError."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
        store = PostgreSQLCredentialStore(encryption_key="test")
        store.encryption_key = None

        with pytest.raises(ValueError, match="Encryption key not configured"):
            store._encrypt_password("password")

    def test_decrypt_empty_returns_empty(self, store):
        """Test decrypting empty string returns empty string."""
        assert store._decrypt_password("") == ""

    def test_decrypt_placeholder_returns_empty(self, store):
        """Test decrypting PLACEHOLDER returns empty string."""
        assert store._decrypt_password("PLACEHOLDER") == ""

    def test_decrypt_without_key_returns_empty(self, store, caplog):
        """Test that decryption without key logs error and returns empty."""
        store.encryption_key = None
        result = store._decrypt_password("encrypted-data")
        assert result == ""
        assert "Encryption key not configured for decryption" in caplog.text

    def test_decrypt_invalid_base64_returns_empty(self, store, caplog):
        """Test that invalid base64 logs error and returns empty."""
        result = store._decrypt_password("not-valid-base64!!!")
        assert result == ""
        assert "Failed to decrypt password" in caplog.text


class TestGet:
    """Tests for get() method."""

    @pytest.fixture
    def store(self, monkeypatch):
        """Create a store for testing."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
        return PostgreSQLCredentialStore(encryption_key="test-key")

    def test_get_found_credentials(self, store):
        """Test getting existing credentials."""
        # Encrypt a test password
        encrypted = store._encrypt_password("vnc-password")

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "hostname": "testhost",
            "username": "testuser",
            "password_encrypted": encrypted,
            "port": 5901,
        }
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.get("testhost")

        assert result is not None
        assert isinstance(result, VNCCredentials)
        assert result.server == "testhost:1"  # port 5901 = display :1
        assert result.password == "vnc-password"
        # Note: close() is not called when credentials are found (early return)

    def test_get_default_port(self, store):
        """Test getting credentials with default port (None -> 5900)."""
        encrypted = store._encrypt_password("password")

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "hostname": "testhost",
            "username": None,
            "password_encrypted": encrypted,
            "port": None,  # Should default to 5900
        }
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.get("testhost")

        assert result is not None
        assert result.server == "testhost:0"  # port 5900 = display :0

    def test_get_not_found(self, store):
        """Test getting non-existent credentials returns None."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.get("nonexistent")

        assert result is None
        mock_conn.close.assert_called_once()

    def test_get_database_error_returns_none(self, store, caplog):
        """Test that database error returns None and logs error."""
        with patch.object(store, "_get_connection", side_effect=Exception("DB Error")):
            result = store.get("testhost")

        assert result is None
        assert "Failed to get credentials from database" in caplog.text


class TestSet:
    """Tests for set() method."""

    @pytest.fixture
    def store(self, monkeypatch):
        """Create a store for testing."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
        return PostgreSQLCredentialStore(encryption_key="test-key")

    def test_set_with_password(self, store):
        """Test storing credentials with password."""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            store.set("testhost", "testhost::5901", "password123")

        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        assert "INSERT INTO vnc_credentials" in call_args[0][0]
        assert call_args[0][1][0] == "testhost"  # hostname
        assert call_args[0][1][2] == 5901  # port
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_set_without_password(self, store):
        """Test storing credentials without password."""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            store.set("testhost", "testhost::5901", None)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][1] == ""  # encrypted_password should be empty

    def test_set_default_port(self, store):
        """Test storing credentials with default port."""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            store.set("testhost", "testhost", "password")

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][2] == 5900  # default port

    def test_set_invalid_port_uses_default(self, store):
        """Test that invalid port falls back to default."""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            store.set("testhost", "testhost::invalid", "password")

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][2] == 5900  # default port due to invalid

    def test_set_database_error_raises(self, store, caplog):
        """Test that database error is re-raised."""
        with patch.object(store, "_get_connection", side_effect=Exception("DB Error")):
            with pytest.raises(Exception, match="DB Error"):
                store.set("testhost", "testhost::5901", "password")

        assert "Failed to store credentials in database" in caplog.text


class TestDelete:
    """Tests for delete() method."""

    @pytest.fixture
    def store(self, monkeypatch):
        """Create a store for testing."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
        return PostgreSQLCredentialStore(encryption_key="test-key")

    def test_delete_existing(self, store):
        """Test deleting existing credentials returns True."""
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.delete("testhost")

        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_delete_not_found(self, store):
        """Test deleting non-existent credentials returns False."""
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.delete("nonexistent")

        assert result is False

    def test_delete_database_error_returns_false(self, store, caplog):
        """Test that database error returns False and logs error."""
        with patch.object(store, "_get_connection", side_effect=Exception("DB Error")):
            result = store.delete("testhost")

        assert result is False
        assert "Failed to delete credentials from database" in caplog.text


class TestListHosts:
    """Tests for list_hosts() method."""

    @pytest.fixture
    def store(self, monkeypatch):
        """Create a store for testing."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
        return PostgreSQLCredentialStore(encryption_key="test-key")

    def test_list_hosts_with_results(self, store):
        """Test listing hosts with results."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("host1",), ("host2",), ("host3",)]
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.list_hosts()

        assert result == ["host1", "host2", "host3"]
        mock_conn.close.assert_called_once()

    def test_list_hosts_empty(self, store):
        """Test listing hosts when database is empty."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.list_hosts()

        assert result == []

    def test_list_hosts_database_error_returns_empty(self, store, caplog):
        """Test that database error returns empty list and logs error."""
        with patch.object(store, "_get_connection", side_effect=Exception("DB Error")):
            result = store.list_hosts()

        assert result == []
        assert "Failed to list hosts from database" in caplog.text


class TestGetMaistackStore:
    """Tests for get_maistack_store factory function."""

    def test_get_maistack_store_returns_postgres_store(self, monkeypatch):
        """Test that factory returns PostgreSQLCredentialStore."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")

        store = get_maistack_store()

        assert isinstance(store, PostgreSQLCredentialStore)

    def test_get_maistack_store_raises_without_database_url(self, monkeypatch):
        """Test that factory raises if DATABASE_URL is not set."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with pytest.raises(ValueError, match="DATABASE_URL environment variable not set"):
            get_maistack_store()


class TestGetConnection:
    """Tests for _get_connection method."""

    def test_get_connection_calls_psycopg2_connect(self, monkeypatch):
        """Test that _get_connection calls psycopg2.connect with database_url."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
        store = PostgreSQLCredentialStore()

        mock_conn = MagicMock()
        with patch(
            "vnc_use.credential_store_postgres.psycopg2.connect", return_value=mock_conn
        ) as mock_connect:
            result = store._get_connection()

        mock_connect.assert_called_once_with("postgresql://user:pass@localhost/db")
        assert result == mock_conn
