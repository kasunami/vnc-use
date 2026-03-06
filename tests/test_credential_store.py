"""Tests for credential_store module."""

import json
import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.vnc_use.credential_store import (
    ChainedStore,
    CredentialStore,
    EnvironmentStore,
    KeyringStore,
    NetrcStore,
    VNCCredentials,
    get_default_store,
)


class TestVNCCredentials:
    """Tests for VNCCredentials class."""

    def test_init_with_server_only(self):
        """Should initialize with server only."""
        creds = VNCCredentials(server="localhost::5901")
        assert creds.server == "localhost::5901"
        assert creds.password is None

    def test_init_with_password(self):
        """Should initialize with server and password."""
        creds = VNCCredentials(server="myhost::5902", password="secret123")
        assert creds.server == "myhost::5902"
        assert creds.password == "secret123"

    def test_repr_without_password(self):
        """Should show None for password in repr."""
        creds = VNCCredentials(server="host1")
        result = repr(creds)
        assert "host1" in result
        assert "None" in result

    def test_repr_with_password(self):
        """Should mask password in repr."""
        creds = VNCCredentials(server="host1", password="secret")
        result = repr(creds)
        assert "host1" in result
        assert "***" in result
        assert "secret" not in result


class TestNetrcStore:
    """Tests for NetrcStore class."""

    def test_init_default_path(self):
        """Should use default path when not specified."""
        store = NetrcStore()
        assert ".vnc_credentials" in store.file_path

    def test_init_custom_path(self, tmp_path: Path):
        """Should use custom path when specified."""
        custom_path = str(tmp_path / "custom_netrc")
        store = NetrcStore(file_path=custom_path)
        assert store.file_path == custom_path

    def test_init_expands_user_path(self):
        """Should expand ~ in path."""
        store = NetrcStore(file_path="~/test_creds")
        assert "~" not in store.file_path
        assert store.file_path.startswith(os.path.expanduser("~"))

    def test_get_returns_none_when_file_missing(self, tmp_path: Path):
        """Should return None when netrc file doesn't exist."""
        store = NetrcStore(file_path=str(tmp_path / "nonexistent"))
        result = store.get("somehost")
        assert result is None

    def test_get_returns_credentials(self, tmp_path: Path):
        """Should return credentials when found."""
        netrc_path = tmp_path / "netrc"
        netrc_path.write_text("machine testhost\nlogin testhost::5901\npassword secret123\n")

        store = NetrcStore(file_path=str(netrc_path))
        result = store.get("testhost")

        assert result is not None
        assert result.server == "testhost::5901"
        assert result.password == "secret123"

    def test_get_returns_none_for_unknown_host(self, tmp_path: Path):
        """Should return None for unknown hostname."""
        netrc_path = tmp_path / "netrc"
        netrc_path.write_text("machine knownhost\nlogin knownhost::5901\npassword secret\n")

        store = NetrcStore(file_path=str(netrc_path))
        result = store.get("unknownhost")
        assert result is None

    def test_get_handles_login_as_server(self, tmp_path: Path):
        """Should use login field as server address."""
        netrc_path = tmp_path / "netrc"
        # In VNC store, login contains the full server address
        netrc_path.write_text("machine testhost\nlogin testhost::5901\npassword secret\n")

        store = NetrcStore(file_path=str(netrc_path))
        result = store.get("testhost")

        assert result is not None
        assert result.server == "testhost::5901"
        assert result.password == "secret"

    def test_get_handles_parse_error(self, tmp_path: Path):
        """Should return None on parse error."""
        netrc_path = tmp_path / "netrc"
        netrc_path.write_text("invalid netrc content @@##")

        store = NetrcStore(file_path=str(netrc_path))
        result = store.get("host")
        assert result is None

    def test_set_creates_file(self, tmp_path: Path):
        """Should create netrc file if not exists."""
        netrc_path = tmp_path / "subdir" / "netrc"
        store = NetrcStore(file_path=str(netrc_path))

        store.set("myhost", "myhost::5901", "mypass")

        assert netrc_path.exists()
        content = netrc_path.read_text()
        assert "machine myhost" in content
        assert "login myhost::5901" in content
        assert "password mypass" in content

    def test_set_appends_to_existing(self, tmp_path: Path):
        """Should append to existing netrc file."""
        netrc_path = tmp_path / "netrc"
        netrc_path.write_text("machine existing\nlogin existing::5901\npassword existingpass\n")

        store = NetrcStore(file_path=str(netrc_path))
        store.set("newhost", "newhost::5902", "newpass")

        content = netrc_path.read_text()
        assert "machine existing" in content
        assert "machine newhost" in content

    def test_set_without_password(self, tmp_path: Path):
        """Should set credentials without password."""
        netrc_path = tmp_path / "netrc"
        store = NetrcStore(file_path=str(netrc_path))

        store.set("host", "host::5901", None)

        content = netrc_path.read_text()
        assert "machine host" in content
        assert "password" not in content

    def test_set_restricts_file_permissions(self, tmp_path: Path):
        """Should set file permissions to 600."""
        netrc_path = tmp_path / "netrc"
        store = NetrcStore(file_path=str(netrc_path))

        store.set("host", "host::5901", "pass")

        # Check permissions (owner read/write only)
        mode = netrc_path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_delete_removes_entry(self, tmp_path: Path):
        """Should remove entry from netrc file."""
        netrc_path = tmp_path / "netrc"
        netrc_path.write_text(
            "machine host1\n"
            "login host1::5901\n"
            "password pass1\n\n"
            "machine host2\n"
            "login host2::5902\n"
            "password pass2\n"
        )

        store = NetrcStore(file_path=str(netrc_path))
        result = store.delete("host1")

        assert result is True
        content = netrc_path.read_text()
        assert "machine host1" not in content
        assert "machine host2" in content

    def test_delete_returns_false_for_missing_file(self, tmp_path: Path):
        """Should return False when file doesn't exist."""
        store = NetrcStore(file_path=str(tmp_path / "nonexistent"))
        result = store.delete("host")
        assert result is False

    def test_list_hosts_returns_hostnames(self, tmp_path: Path):
        """Should return list of hostnames."""
        netrc_path = tmp_path / "netrc"
        netrc_path.write_text(
            "machine host1\n"
            "login host1::5901\n"
            "password pass1\n\n"
            "machine host2\n"
            "login host2::5902\n"
            "password pass2\n"
        )

        store = NetrcStore(file_path=str(netrc_path))
        hosts = store.list_hosts()

        assert len(hosts) == 2
        assert "host1" in hosts
        assert "host2" in hosts

    def test_list_hosts_returns_empty_for_missing_file(self, tmp_path: Path):
        """Should return empty list when file missing."""
        store = NetrcStore(file_path=str(tmp_path / "nonexistent"))
        hosts = store.list_hosts()
        assert hosts == []

    def test_list_hosts_returns_empty_on_parse_error(self, tmp_path: Path):
        """Should return empty list on parse error."""
        netrc_path = tmp_path / "netrc"
        netrc_path.write_text("invalid @@##")

        store = NetrcStore(file_path=str(netrc_path))
        hosts = store.list_hosts()
        assert hosts == []


class TestKeyringStore:
    """Tests for KeyringStore class."""

    def test_init_imports_keyring(self):
        """Should import keyring on init."""
        with patch.dict("sys.modules", {"keyring": MagicMock()}):
            store = KeyringStore()
            assert store.keyring is not None

    def test_init_raises_on_missing_keyring(self):
        """Should raise ImportError when keyring not available."""
        with patch.dict("sys.modules", {"keyring": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="keyring package required"):
                    KeyringStore()

    def test_get_returns_credentials(self):
        """Should return credentials from keyring."""
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = json.dumps(
            {"server": "testhost::5901", "password": "secret"}
        )

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring
            result = store.get("testhost")

        assert result is not None
        assert result.server == "testhost::5901"
        assert result.password == "secret"
        mock_keyring.get_password.assert_called_once_with("vnc-use", "testhost")

    def test_get_returns_none_when_not_found(self):
        """Should return None when no credentials found."""
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = None

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring
            result = store.get("unknownhost")

        assert result is None

    def test_get_returns_none_on_error(self):
        """Should return None on keyring error."""
        mock_keyring = MagicMock()
        mock_keyring.get_password.side_effect = Exception("Keyring error")

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring
            result = store.get("host")

        assert result is None

    def test_get_uses_hostname_when_server_missing(self):
        """Should use hostname as server when not in stored data."""
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = json.dumps({"password": "secret"})

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring
            result = store.get("myhost")

        assert result is not None
        assert result.server == "myhost"

    def test_set_stores_credentials(self):
        """Should store credentials in keyring."""
        mock_keyring = MagicMock()

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring
            store.set("testhost", "testhost::5901", "secret")

        mock_keyring.set_password.assert_called_once()
        call_args = mock_keyring.set_password.call_args
        assert call_args[0][0] == "vnc-use"
        assert call_args[0][1] == "testhost"
        stored_data = json.loads(call_args[0][2])
        assert stored_data["server"] == "testhost::5901"
        assert stored_data["password"] == "secret"

    def test_set_raises_on_error(self):
        """Should raise exception on keyring error."""
        mock_keyring = MagicMock()
        mock_keyring.set_password.side_effect = Exception("Store failed")

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring

            with pytest.raises(Exception, match="Store failed"):
                store.set("host", "host::5901", "pass")

    def test_delete_removes_credentials(self):
        """Should delete credentials from keyring."""
        mock_keyring = MagicMock()
        mock_keyring.errors = MagicMock()
        mock_keyring.errors.PasswordDeleteError = type("PasswordDeleteError", (Exception,), {})

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring
            result = store.delete("testhost")

        assert result is True
        mock_keyring.delete_password.assert_called_once_with("vnc-use", "testhost")

    def test_delete_returns_false_when_not_found(self):
        """Should return False when password not found."""
        mock_keyring = MagicMock()
        mock_keyring.errors = MagicMock()
        mock_keyring.errors.PasswordDeleteError = type("PasswordDeleteError", (Exception,), {})
        mock_keyring.delete_password.side_effect = mock_keyring.errors.PasswordDeleteError()

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring
            result = store.delete("unknownhost")

        assert result is False

    def test_delete_returns_false_on_other_error(self):
        """Should return False on other errors."""
        mock_keyring = MagicMock()
        mock_keyring.errors = MagicMock()
        mock_keyring.errors.PasswordDeleteError = type("PasswordDeleteError", (Exception,), {})
        mock_keyring.delete_password.side_effect = Exception("Other error")

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring
            result = store.delete("host")

        assert result is False

    def test_list_hosts_returns_empty(self):
        """Should return empty list (enumeration not supported)."""
        mock_keyring = MagicMock()

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeyringStore()
            store.keyring = mock_keyring
            hosts = store.list_hosts()

        assert hosts == []


class TestEnvironmentStore:
    """Tests for EnvironmentStore class."""

    def test_get_returns_credentials_from_env(self, monkeypatch):
        """Should return credentials from environment variables."""
        monkeypatch.setenv("VNC_SERVER", "envhost::5901")
        monkeypatch.setenv("VNC_PASSWORD", "envpass")

        store = EnvironmentStore()
        result = store.get("anyhost")

        assert result is not None
        assert result.server == "envhost::5901"
        assert result.password == "envpass"

    def test_get_returns_none_when_server_not_set(self, monkeypatch):
        """Should return None when VNC_SERVER not set."""
        monkeypatch.delenv("VNC_SERVER", raising=False)
        monkeypatch.delenv("VNC_PASSWORD", raising=False)

        store = EnvironmentStore()
        result = store.get("host")

        assert result is None

    def test_get_works_without_password(self, monkeypatch):
        """Should work when only VNC_SERVER is set."""
        monkeypatch.setenv("VNC_SERVER", "host::5901")
        monkeypatch.delenv("VNC_PASSWORD", raising=False)

        store = EnvironmentStore()
        result = store.get("anyhost")

        assert result is not None
        assert result.server == "host::5901"
        assert result.password is None

    def test_get_ignores_hostname(self, monkeypatch):
        """Should ignore hostname parameter."""
        monkeypatch.setenv("VNC_SERVER", "fixed::5901")

        store = EnvironmentStore()
        result1 = store.get("host1")
        result2 = store.get("host2")

        assert result1 is not None
        assert result2 is not None
        assert result1.server == result2.server == "fixed::5901"

    def test_set_raises_not_implemented(self):
        """Should raise NotImplementedError on set."""
        store = EnvironmentStore()
        with pytest.raises(NotImplementedError, match="does not support set"):
            store.set("host", "host::5901", "pass")

    def test_delete_raises_not_implemented(self):
        """Should raise NotImplementedError on delete."""
        store = EnvironmentStore()
        with pytest.raises(NotImplementedError, match="does not support delete"):
            store.delete("host")

    def test_list_hosts_returns_server_when_set(self, monkeypatch):
        """Should return VNC_SERVER when set."""
        monkeypatch.setenv("VNC_SERVER", "myhost::5901")

        store = EnvironmentStore()
        hosts = store.list_hosts()

        assert hosts == ["myhost::5901"]

    def test_list_hosts_returns_empty_when_not_set(self, monkeypatch):
        """Should return empty list when VNC_SERVER not set."""
        monkeypatch.delenv("VNC_SERVER", raising=False)

        store = EnvironmentStore()
        hosts = store.list_hosts()

        assert hosts == []


class TestChainedStore:
    """Tests for ChainedStore class."""

    def test_get_returns_first_match(self):
        """Should return credentials from first store that has them."""
        store1 = MagicMock(spec=CredentialStore)
        store1.get.return_value = None

        store2 = MagicMock(spec=CredentialStore)
        store2.get.return_value = VNCCredentials("host::5901", "pass")

        store3 = MagicMock(spec=CredentialStore)

        chained = ChainedStore([store1, store2, store3])
        result = chained.get("host")

        assert result is not None
        assert result.server == "host::5901"
        store1.get.assert_called_once_with("host")
        store2.get.assert_called_once_with("host")
        store3.get.assert_not_called()

    def test_get_returns_none_when_not_found(self):
        """Should return None when no store has credentials."""
        store1 = MagicMock(spec=CredentialStore)
        store1.get.return_value = None

        store2 = MagicMock(spec=CredentialStore)
        store2.get.return_value = None

        chained = ChainedStore([store1, store2])
        result = chained.get("unknown")

        assert result is None

    def test_set_uses_first_writable_store(self):
        """Should set credentials in first non-EnvironmentStore."""
        env_store = EnvironmentStore()
        writable_store = MagicMock(spec=CredentialStore)

        chained = ChainedStore([env_store, writable_store])
        chained.set("host", "host::5901", "pass")

        writable_store.set.assert_called_once_with("host", "host::5901", "pass")

    def test_set_raises_when_no_writable_store(self):
        """Should raise when all stores are EnvironmentStore."""
        env_store = EnvironmentStore()

        chained = ChainedStore([env_store])

        with pytest.raises(RuntimeError, match="No writable credential store"):
            chained.set("host", "host::5901", "pass")

    def test_delete_deletes_from_all_stores(self):
        """Should try to delete from all stores."""
        store1 = MagicMock(spec=CredentialStore)
        store1.delete.return_value = True

        store2 = MagicMock(spec=CredentialStore)
        store2.delete.return_value = False

        chained = ChainedStore([store1, store2])
        result = chained.delete("host")

        assert result is True
        store1.delete.assert_called_once_with("host")
        store2.delete.assert_called_once_with("host")

    def test_delete_returns_false_when_nothing_deleted(self):
        """Should return False when no store had credentials."""
        store1 = MagicMock(spec=CredentialStore)
        store1.delete.return_value = False

        chained = ChainedStore([store1])
        result = chained.delete("unknown")

        assert result is False

    def test_delete_handles_not_implemented(self):
        """Should continue when store raises NotImplementedError."""
        env_store = EnvironmentStore()
        writable_store = MagicMock(spec=CredentialStore)
        writable_store.delete.return_value = True

        chained = ChainedStore([env_store, writable_store])
        result = chained.delete("host")

        assert result is True

    def test_list_hosts_combines_all_stores(self):
        """Should combine hosts from all stores."""
        store1 = MagicMock(spec=CredentialStore)
        store1.list_hosts.return_value = ["host1", "host2"]

        store2 = MagicMock(spec=CredentialStore)
        store2.list_hosts.return_value = ["host2", "host3"]

        chained = ChainedStore([store1, store2])
        hosts = chained.list_hosts()

        assert sorted(hosts) == ["host1", "host2", "host3"]

    def test_list_hosts_deduplicates(self):
        """Should not return duplicate hostnames."""
        store1 = MagicMock(spec=CredentialStore)
        store1.list_hosts.return_value = ["host1"]

        store2 = MagicMock(spec=CredentialStore)
        store2.list_hosts.return_value = ["host1"]

        chained = ChainedStore([store1, store2])
        hosts = chained.list_hosts()

        assert hosts == ["host1"]


class TestGetDefaultStore:
    """Tests for get_default_store function."""

    def test_returns_chained_store(self):
        """Should return a ChainedStore."""
        result = get_default_store()
        assert isinstance(result, ChainedStore)

    def test_includes_netrc_store(self):
        """Should include NetrcStore in chain."""
        result = cast(ChainedStore, get_default_store())
        stores = cast(list[Any], result.stores)
        store_types = [type(s).__name__ for s in stores]
        assert "NetrcStore" in store_types

    def test_includes_environment_store(self):
        """Should include EnvironmentStore in chain."""
        result = cast(ChainedStore, get_default_store())
        stores = cast(list[Any], result.stores)
        store_types = [type(s).__name__ for s in stores]
        assert "EnvironmentStore" in store_types

    def test_keyring_included_when_available(self):
        """Should include KeyringStore when keyring available."""
        mock_keyring = MagicMock()
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            result = cast(ChainedStore, get_default_store())
            stores = cast(list[Any], result.stores)
            store_types = [type(s).__name__ for s in stores]
            assert "KeyringStore" in store_types

    def test_works_without_keyring(self):
        """Should work when keyring not available."""
        with patch("src.vnc_use.credential_store.KeyringStore") as mock_keyring_cls:
            mock_keyring_cls.side_effect = ImportError("No keyring")
            result = cast(ChainedStore, get_default_store())
            assert isinstance(result, ChainedStore)
            # Should still have at least NetrcStore and EnvironmentStore
            assert len(result.stores) >= 2
