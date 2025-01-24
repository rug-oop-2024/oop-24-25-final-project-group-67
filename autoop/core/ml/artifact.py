from pydantic import BaseModel, Field
import base64

class Artifact(BaseModel):
    """
    Base class for representing an artifact.

    Attributes:
        asset_path: File path where the artifact is stored.
        version: Version of the artifact.
        data: content of the artifact.
        metadata: Additional metadata.
        type: The type of artifact.
    """
    asset_path: str
    version: str = "1.0.0"
    data: bytes
    metadata: dict = Field(default_factory=dict)
    type: str

    def get_id(self) -> str:
        """
        Generates ID for artifact based on the asset path and version.

        Returns:
            str: ID for the artifact.
        """
        encoded_path = base64.urlsafe_b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """
        Returns binary data stored in the artifact.

        Returns:
            bytes: The binary data of the artifact.
        """
        return self.data

    def save(self, new_data: bytes) -> None:
        """
        Updates the artifact's data and metadata for persistence.

        Args:
            new_data (bytes): The new binary data to save.
        """
        self.data = new_data
