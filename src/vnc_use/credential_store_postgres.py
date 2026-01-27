"""PostgreSQL credential store for VNC server authentication in maistack.

Replaces keyring/netrc stores with database-backed credential management.
"""

import base64
import logging
import os

import psycopg2
from psycopg2.extras import RealDictCursor

from .credential_store import CredentialStore, VNCCredentials

logger = logging.getLogger(__name__)


class PostgreSQLCredentialStore(CredentialStore):
    """Credential store using PostgreSQL database.

    Stores VNC credentials in the maistack PostgreSQL database with encryption.
    Credentials are encrypted using application-level encryption before storage.
    """

    def __init__(self, database_url: str | None = None, encryption_key: str | None = None):
        """Initialize PostgreSQL credential store.

        Args:
            database_url: PostgreSQL connection URL (defaults to DATABASE_URL env var)
            encryption_key: Encryption key for passwords (defaults to VNC_ENCRYPTION_KEY env var)
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        self.encryption_key = encryption_key or os.getenv(
            "VNC_ENCRYPTION_KEY", "default-key-change-me"
        )
        if self.encryption_key == "default-key-change-me":
            logger.warning("Using default encryption key! Set VNC_ENCRYPTION_KEY for production")

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.database_url)

    def _encrypt_password(self, password: str) -> str:
        """Simple XOR-based encryption for passwords.

        Note: For production, use proper encryption like Fernet or pgcrypto.
        """
        if not password:
            return ""
        if not self.encryption_key:
            raise ValueError("Encryption key not configured")
        key_bytes = self.encryption_key.encode()
        pwd_bytes = password.encode()
        encrypted = bytes(
            pwd_bytes[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(pwd_bytes))
        )
        return base64.b64encode(encrypted).decode("utf-8")

    def _decrypt_password(self, encrypted: str) -> str:
        """Decrypt password."""
        if not encrypted or encrypted == "PLACEHOLDER":
            return ""
        if not self.encryption_key:
            logger.error("Encryption key not configured for decryption")
            return ""
        try:
            key_bytes = self.encryption_key.encode()
            encrypted_bytes = base64.b64decode(encrypted.encode("utf-8"))
            decrypted = bytes(
                encrypted_bytes[i] ^ key_bytes[i % len(key_bytes)]
                for i in range(len(encrypted_bytes))
            )
            return decrypted.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to decrypt password: {e}")
            return ""

    def get(self, hostname: str) -> VNCCredentials | None:
        """Get credentials from PostgreSQL database.

        Args:
            hostname: VNC server hostname or address

        Returns:
            VNCCredentials if found, None otherwise
        """
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT hostname, username, password_encrypted, port FROM vnc_credentials WHERE hostname = %s",
                    (hostname,),
                )
                row = cur.fetchone()

                if row:
                    password = self._decrypt_password(row["password_encrypted"])
                    # Construct server address using display number format (vncdotool requirement)
                    # Port 5900 = display :0, 5901 = display :1, etc.
                    port = row["port"] or 5900
                    display_num = port - 5900
                    server = f"{hostname}:{display_num}"

                    logger.debug(f"Found credentials for {hostname} (display :{display_num})")
                    return VNCCredentials(server=server, password=password)

            conn.close()
            return None

        except Exception as e:
            logger.error(f"Failed to get credentials from database: {e}")
            return None

    def set(self, hostname: str, server: str, password: str | None = None) -> None:
        """Store credentials in PostgreSQL database.

        Args:
            hostname: VNC server hostname or address
            server: Full VNC server address (e.g., "hostname::5901")
            password: VNC password (optional)
        """
        try:
            # Parse port from server address
            port = 5900
            if "::" in server:
                port_str = server.split("::")[-1]
                try:
                    port = int(port_str)
                except ValueError:
                    pass

            encrypted_password = self._encrypt_password(password) if password else ""

            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO vnc_credentials (hostname, password_encrypted, port, description)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (hostname)
                    DO UPDATE SET
                        password_encrypted = EXCLUDED.password_encrypted,
                        port = EXCLUDED.port,
                        updated_at = NOW()
                    """,
                    (hostname, encrypted_password, port, "Added via MCP server"),
                )
                conn.commit()

            conn.close()
            logger.info(f"Stored credentials for {hostname} in database")

        except Exception as e:
            logger.error(f"Failed to store credentials in database: {e}")
            raise

    def delete(self, hostname: str) -> bool:
        """Delete credentials from PostgreSQL database.

        Args:
            hostname: VNC server hostname or address

        Returns:
            True if credentials were deleted, False if not found
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM vnc_credentials WHERE hostname = %s", (hostname,))
                deleted = cur.rowcount > 0
                conn.commit()

            conn.close()

            if deleted:
                logger.info(f"Deleted credentials for {hostname}")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete credentials from database: {e}")
            return False

    def list_hosts(self) -> list[str]:
        """List all hostnames in PostgreSQL database.

        Returns:
            List of hostname strings
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT hostname FROM vnc_credentials ORDER BY hostname")
                hosts = [row[0] for row in cur.fetchall()]

            conn.close()
            return hosts

        except Exception as e:
            logger.error(f"Failed to list hosts from database: {e}")
            return []


def get_maistack_store() -> CredentialStore:
    """Get PostgreSQL credential store for maistack.

    Returns:
        PostgreSQLCredentialStore configured for maistack
    """
    return PostgreSQLCredentialStore()
