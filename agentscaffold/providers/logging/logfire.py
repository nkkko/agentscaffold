"""Logging provider implementation using LogFire."""

import os
import json
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable

# Try to import optional dependencies
try:
    import logfire
    HAS_LOGFIRE = True
except ImportError:
    logfire = None
    HAS_LOGFIRE = False

class LogFireProvider:
    """Logging provider using LogFire for observability."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        service_name: str = "agent",
        environment: str = "development",
        enable_console: bool = True,
        log_level: int = logging.INFO,
        **kwargs
    ):
        """
        Initialize the LogFire logging provider.
        
        Args:
            api_key: LogFire API key (defaults to LOGFIRE_API_KEY env var)
            service_name: Name of the service for logging
            environment: Environment (development, staging, production)
            enable_console: Whether to log to console as well
            log_level: Logging level (default: INFO)
            **kwargs: Additional parameters for LogFire
        """
        if not HAS_LOGFIRE:
            print("Warning: logfire is not installed. Only basic console logging will be available.")
            self._setup_basic_logging(service_name, log_level)
            self.service_name = service_name
            self.conversation_id = None
            self.conversation_context = {}
            self.start_time = None
            return
        
        self.api_key = api_key or os.environ.get("LOGFIRE_API_KEY")
        if not self.api_key:
            print("Warning: LogFire API key is required. Set LOGFIRE_API_KEY environment variable or pass as api_key.")
            print("Only basic console logging will be available.")
            self._setup_basic_logging(service_name, log_level)
            self.service_name = service_name
            self.conversation_id = None
            self.conversation_context = {}
            self.start_time = None
            return
        
        self.service_name = service_name
        self.environment = environment
        
        # Initialize LogFire - trying multiple initialization approaches
        try:
            self._initialize_logfire(service_name, environment, enable_console, log_level, **kwargs)
        except Exception as e:
            print(f"Error initializing LogFire: {e}")
            self._setup_basic_logging(service_name, log_level)
        
        # Create a logger for the agent
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(log_level)
        
        # Keep track of conversation context
        self.conversation_id = None
        self.conversation_context = {}
        self.start_time = None
    
    def _initialize_logfire(self, service_name, environment, enable_console, log_level, **kwargs):
        """Initialize LogFire using multiple attempts with different API patterns."""
        # Attempt 1: Latest LogFire API with configure method and token parameter
        if hasattr(logfire, 'configure'):
            try:
                # Clean kwargs to avoid unexpected argument errors
                config_kwargs = kwargs.copy()
                for k in ['token', 'api_key', 'api_token', 'service', 'service_name', 'environment', 'console', 'level']:
                    if k in config_kwargs:
                        del config_kwargs[k]
                
                # Try with token parameter
                logfire.configure(
                    token=self.api_key,
                    service=service_name,
                    environment=environment,
                    console=enable_console,
                    level=log_level,
                    **config_kwargs
                )
                print(f"Initialized LogFire using configure() with token parameter")
                return
            except (TypeError, ValueError) as e:
                print(f"LogFire configure attempt 1 failed: {e}")
                
                # Attempt 2: Try with api_token parameter
                try:
                    logfire.configure(
                        api_token=self.api_key,
                        service=service_name,
                        environment=environment,
                        console=enable_console,
                        level=log_level,
                        **config_kwargs
                    )
                    print(f"Initialized LogFire using configure() with api_token parameter")
                    return
                except (TypeError, ValueError) as e:
                    print(f"LogFire configure attempt 2 failed: {e}")
                    
                    # Attempt 3: Try with minimal parameters
                    try:
                        logfire.configure(
                            token=self.api_key
                        )
                        print(f"Initialized LogFire using configure() with minimal parameters")
                        return
                    except (TypeError, ValueError) as e:
                        print(f"LogFire configure attempt 3 failed: {e}")
        
        # Attempt 4: Older LogFire API with init method
        if hasattr(logfire, 'init'):
            try:
                # Clean kwargs to avoid unexpected argument errors
                init_kwargs = kwargs.copy()
                for k in ['token', 'api_key', 'api_token', 'service_name', 'environment', 'console', 'level']:
                    if k in init_kwargs:
                        del init_kwargs[k]
                        
                logfire.init(
                    token=self.api_key,
                    service_name=service_name,
                    environment=environment,
                    console=enable_console,
                    level=log_level,
                    **init_kwargs
                )
                print(f"Initialized LogFire using init() method")
                return
            except (TypeError, ValueError) as e:
                print(f"LogFire init attempt failed: {e}")
        
        # Attempt 5: Check for alternate initialization methods
        if hasattr(logfire, 'setup'):
            try:
                logfire.setup(
                    api_key=self.api_key,
                    service=service_name,
                    **kwargs
                )
                print(f"Initialized LogFire using setup() method")
                return
            except (TypeError, ValueError) as e:
                print(f"LogFire setup attempt failed: {e}")
        
        # If all attempts fail, try the most minimal approach
        try:
            # Find any method that might be for initialization
            for method_name in dir(logfire):
                if method_name.lower() in ['configure', 'init', 'setup', 'initialize']:
                    method = getattr(logfire, method_name)
                    if callable(method):
                        try:
                            # Try with just the API key
                            method(self.api_key)
                            print(f"Initialized LogFire using {method_name}() with just API key")
                            return
                        except Exception:
                            pass
        except Exception:
            pass
        
        # If we reach here, all initialization attempts failed
        raise ValueError("Could not initialize LogFire with any known method")
    
    def _setup_basic_logging(self, service_name: str, log_level: int) -> None:
        """Set up basic console logging as fallback."""
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(log_level)
        
        # Check if handler already exists to avoid duplicates
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        print(f"Set up basic console logging for {service_name}")
    
    def start_conversation(self, user_id: Optional[str] = None) -> str:
        """
        Start a new conversation and return the conversation ID.
        
        Args:
            user_id: Optional identifier for the user
            
        Returns:
            Conversation ID
        """
        self.conversation_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.conversation_context = {
            "conversation_id": self.conversation_id,
            "user_id": user_id or "anonymous",
            "start_time": self.start_time,
            "messages": []
        }
        
        # Log conversation start
        if HAS_LOGFIRE:
            try:
                logfire.info(
                    "conversation_started",
                    conversation_id=self.conversation_id,
                    user_id=self.conversation_context["user_id"]
                )
            except Exception as e:
                self.logger.info(f"Conversation started: {self.conversation_id} (User: {self.conversation_context['user_id']})")
                self.logger.error(f"Error logging to LogFire: {e}")
        else:
            self.logger.info(f"Conversation started: {self.conversation_id} (User: {self.conversation_context['user_id']})")
        
        return self.conversation_id
    
    def log_user_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a user message in the current conversation.
        
        Args:
            message: User message
            metadata: Optional metadata
        """
        if not self.conversation_id:
            self.start_conversation()
            
        # Add message to conversation context
        msg_data = {
            "role": "user",
            "content": message,
            "timestamp": time.time()
        }
        if metadata:
            msg_data["metadata"] = metadata
            
        self.conversation_context["messages"].append(msg_data)
        
        # Log with LogFire if available
        if HAS_LOGFIRE:
            try:
                # Some environments might require suppressing logfire warnings
                os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
                
                logfire.info(
                    "user_message",
                    conversation_id=self.conversation_id,
                    message=message,
                    **({} if not metadata else {"metadata": json.dumps(metadata)})
                )
            except Exception as e:
                self.logger.info(f"User message: {message[:100]}...")
                self.logger.error(f"Error logging to LogFire: {e}")
        else:
            self.logger.info(f"User message: {message[:100]}...")
    
    def log_agent_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an agent message in the current conversation.
        
        Args:
            message: Agent message
            metadata: Optional metadata
        """
        if not self.conversation_id:
            self.start_conversation()
            
        # Add message to conversation context
        msg_data = {
            "role": "agent",
            "content": message,
            "timestamp": time.time()
        }
        if metadata:
            msg_data["metadata"] = metadata
            
        self.conversation_context["messages"].append(msg_data)
        
        # Log with LogFire if available
        if HAS_LOGFIRE:
            try:
                # Some environments might require suppressing logfire warnings
                os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
                
                logfire.info(
                    "agent_message",
                    conversation_id=self.conversation_id,
                    message=message[:200] + ("..." if len(message) > 200 else ""),
                    **({} if not metadata else {"metadata": json.dumps(metadata)})
                )
            except Exception as e:
                self.logger.info(f"Agent message: {message[:100]}...")
                self.logger.error(f"Error logging to LogFire: {e}")
        else:
            self.logger.info(f"Agent message: {message[:100]}...")
    
    def log_function_call(
        self, 
        function_name: str, 
        arguments: Dict[str, Any],
        result: Any = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log a function call by the agent.
        
        Args:
            function_name: Name of the function called
            arguments: Arguments passed to the function
            result: Optional result of the function call
            error: Optional error message if the function failed
        """
        # Calculate duration
        duration = None
        if self.start_time:
            duration = time.time() - self.start_time
        
        # Log with LogFire if available
        if HAS_LOGFIRE:
            try:
                # Some environments might require suppressing logfire warnings
                os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
                
                logfire.info(
                    "function_call",
                    conversation_id=self.conversation_id or "unknown",
                    function_name=function_name,
                    arguments=json.dumps(arguments) if arguments else None,
                    result=json.dumps(result) if result is not None else None,
                    error=error,
                    duration=duration
                )
            except Exception as e:
                self.logger.info(f"Function call: {function_name}")
                self.logger.error(f"Error logging to LogFire: {e}")
        else:
            self.logger.info(f"Function call: {function_name}")
            if error:
                self.logger.error(f"Function error: {error}")
    
    def log_error(self, error: Union[str, Exception], context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error encountered by the agent.
        
        Args:
            error: Error message or exception
            context: Optional context information
        """
        error_msg = str(error)
        error_type = type(error).__name__ if isinstance(error, Exception) else "Error"
        
        # Log with LogFire if available
        if HAS_LOGFIRE:
            try:
                # Some environments might require suppressing logfire warnings
                os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
                
                logfire.error(
                    "agent_error",
                    conversation_id=self.conversation_id or "unknown",
                    error_message=error_msg,
                    error_type=error_type,
                    **({} if not context else {"context": json.dumps(context)})
                )
            except Exception as e:
                self.logger.error(f"Agent error: {error_msg}")
                self.logger.error(f"Error logging to LogFire: {e}")
        else:
            self.logger.error(f"Agent error: {error_msg}")
            if context:
                self.logger.debug(f"Error context: {json.dumps(context)}")
    
    def end_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        End the current conversation and log summary.
        
        Args:
            metadata: Optional metadata about the conversation
        """
        if not self.conversation_id:
            return
            
        # Calculate conversation duration
        duration = None
        if self.start_time:
            duration = time.time() - self.start_time
            
        # Count messages by role
        user_messages = sum(1 for msg in self.conversation_context["messages"] if msg["role"] == "user")
        agent_messages = sum(1 for msg in self.conversation_context["messages"] if msg["role"] == "agent")
        
        # Log with LogFire if available
        if HAS_LOGFIRE:
            try:
                # Some environments might require suppressing logfire warnings
                os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
                
                logfire.info(
                    "conversation_ended",
                    conversation_id=self.conversation_id,
                    user_id=self.conversation_context.get("user_id", "anonymous"),
                    duration=duration,
                    user_message_count=user_messages,
                    agent_message_count=agent_messages,
                    **({} if not metadata else metadata)
                )
            except Exception as e:
                self.logger.info(f"Conversation ended: {self.conversation_id} (Duration: {duration:.2f}s)")
                self.logger.error(f"Error logging to LogFire: {e}")
        else:
            self.logger.info(f"Conversation ended: {self.conversation_id} (Duration: {duration:.2f}s)")
        
        # Reset conversation context
        self.conversation_id = None
        self.conversation_context = {}
        self.start_time = None
    
    def create_trace(
        self, 
        operation_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Create a trace for an operation using the with statement.
        
        Args:
            operation_name: Name of the operation
            context: Optional context information
            
        Returns:
            Context manager for the trace
        """
        # Create a context manager for tracing
        class TraceContextManager:
            def __init__(self, provider, name, ctx):
                self.provider = provider
                self.name = name
                self.context = ctx or {}
                self.start_time = None
                self.span = None
            
            def __enter__(self):
                self.start_time = time.time()
                # Start the span if LogFire is available
                if HAS_LOGFIRE:
                    try:
                        # Some environments might require suppressing logfire warnings
                        os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
                        
                        if hasattr(logfire, 'start_span'):
                            self.span = logfire.start_span(
                                name=self.name,
                                attributes={
                                    "conversation_id": self.provider.conversation_id or "unknown",
                                    **self.context
                                }
                            )
                    except Exception as e:
                        self.provider.logger.error(f"Error starting LogFire span: {e}")
                
                self.provider.logger.info(f"Operation started: {self.name}")
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                
                # Log exception if any
                if exc_type is not None:
                    if HAS_LOGFIRE:
                        try:
                            # Some environments might require suppressing logfire warnings
                            os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
                            
                            logfire.exception(
                                f"{self.name}_error",
                                error=str(exc_val),
                                error_type=exc_type.__name__,
                                duration=duration
                            )
                        except Exception as e:
                            self.provider.logger.error(f"Operation error: {exc_val}")
                            self.provider.logger.error(f"Error logging to LogFire: {e}")
                    else:
                        self.provider.logger.error(f"Operation error: {exc_val}")
                
                # End the span
                if self.span and hasattr(self.span, 'end'):
                    try:
                        self.span.end()
                    except Exception as e:
                        self.provider.logger.error(f"Error ending LogFire span: {e}")
                
                # Log completion
                if HAS_LOGFIRE:
                    try:
                        # Some environments might require suppressing logfire warnings
                        os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
                        
                        logfire.info(
                            f"{self.name}_completed",
                            duration=duration,
                            conversation_id=self.provider.conversation_id or "unknown",
                            success=(exc_type is None)
                        )
                    except Exception as e:
                        self.provider.logger.info(f"Operation completed: {self.name} (Duration: {duration:.2f}s)")
                        self.provider.logger.error(f"Error logging to LogFire: {e}")
                else:
                    self.provider.logger.info(f"Operation completed: {self.name} (Duration: {duration:.2f}s)")
                
                return False  # Don't suppress exceptions
        
        return TraceContextManager(self, operation_name, context)