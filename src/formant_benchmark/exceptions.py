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
    """Raised when tracker execution cannot start or safely continue."""


class UnsupportedTrackerConfigurationError(ConfigurationError):
    """Raised when a tracker cannot execute a requested input configuration."""


class PredictionRunError(FormantBenchmarkError):
    """Base class for prediction-run persistence and compatibility failures."""


class PredictionRunAlreadyExistsError(PredictionRunError):
    """Raised when a new run targets an existing destination."""


class PredictionRunValidationError(PredictionRunError):
    """Raised when prediction artifacts are structurally invalid."""


class ResumeCompatibilityError(PredictionRunError):
    """Raised when resume settings do not match an existing run."""


class EvaluationError(FormantBenchmarkError):
    """Base class for evaluation compatibility and persistence failures."""


class EvaluationCompatibilityError(EvaluationError):
    """Raised when a prepared dataset and PredictionRun cannot be scored together."""


class EvaluationAlreadyExistsError(EvaluationError):
    """Raised when evaluation output already exists without explicit overwrite."""


class EvaluationValidationError(EvaluationError):
    """Raised when persisted evaluation artifacts are structurally invalid."""
