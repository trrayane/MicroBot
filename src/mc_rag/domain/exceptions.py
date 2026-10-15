"""Domain-level exception hierarchy.

These exceptions are raised by the domain and application layers and are
translated into transport-specific responses at the API boundary. Keeping them
independent of any web framework or third-party library lets the core logic
stay testable and portable.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for every error raised by the domain layer."""


class DocumentError(DomainError):
    """Base class for document-related failures."""


class DocumentNotFoundError(DocumentError):
    """Raised when a requested document does not exist on disk."""


class UnsupportedFormatError(DocumentError):
    """Raised when a document's format has no registered parser."""


class ParsingError(DocumentError):
    """Raised when a document is present but cannot be parsed."""


class EmptyDocumentError(ParsingError):
    """Raised when a document contains no extractable text."""


class CorruptDocumentError(ParsingError):
    """Raised when a document is malformed or otherwise unreadable."""


class DocumentTooLargeError(ParsingError):
    """Raised when a document exceeds a configured structural limit (e.g. rows)."""


class UnsafeURLError(ParsingError):
    """Raised when a URL points at a blocked scheme, host or address."""


class LexicalIndexError(DomainError):
    """Raised when the lexical (BM25) index cannot be accessed or queried."""


class EncryptedDocumentError(ParsingError):
    """Raised when a document is encrypted and cannot be decrypted."""


class EmbeddingError(DomainError):
    """Raised when text embedding generation fails."""


class VectorStoreError(DomainError):
    """Raised when the vector store cannot be accessed or queried."""


class LLMError(DomainError):
    """Base class for failures talking to the language model."""


class LLMRateLimitError(LLMError):
    """Raised when the provider rejects a call because a quota was reached."""


class LLMUnavailableError(LLMError):
    """Raised when the model cannot produce a usable answer (network, 5xx, bad payload)."""


class AuthenticationError(DomainError):
    """Raised when a credential is present but invalid."""


class InvalidInput(DomainError):
    """Raised when user input fails validation at the API boundary."""


class UpstreamUnavailable(DomainError):
    """Raised when an external service (Discord, OpenRouter…) is unreachable."""


class ConversationNotFoundError(DomainError):
    """Raised when a conversation does not exist or is not accessible to the caller."""