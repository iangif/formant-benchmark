"""Application-level exceptions with stable, user-facing semantics."""


class FormantBenchmarkError(Exception):
    """Base class for expected benchmark application failures."""


class ConfigurationError(FormantBenchmarkError):
    """Raised when configuration cannot be interpreted safely."""


class ConfigurationConflictError(ConfigurationError):
    """Raised when equally specific parameter rules conflict."""


class DatasetValidationError(FormantBenchmarkError):
    """Raised when prepared dataset structure is invalid."""


class DatasetAlreadyExistsError(FormantBenchmarkError):
    """Raised when a prepared-dataset destination already exists."""


class DatasetFingerprintMismatchError(FormantBenchmarkError):
    """Raised when persisted content does not match its recorded fingerprint."""


class RegistryError(FormantBenchmarkError):
    """Base class for registry lookup/registration failures."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a registry name is registered more than once."""


class UnknownRegistrationError(RegistryError):
    """Raised when a requested registry entry does not exist."""


class UnsupportedScopeError(FormantBenchmarkError):
    """Raised when an evaluation scope exists architecturally but is unavailable."""


class UnsupportedVoicedFeatureError(FormantBenchmarkError):
    """Raised when V1 voiced preparation/tracking functionality is requested."""


class IncompatibleFormantsError(FormantBenchmarkError):
    """Raised when an operation has no compatible non-empty formant set."""


class TrackerExecutionError(FormantBenchmarkError):
    """Reserved for tracker execution failures in later deliverables."""
