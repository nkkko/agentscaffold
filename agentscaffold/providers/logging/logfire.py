import os
import json
import uuid
import time
import logging
import traceback
from typing import Dict, Any, Optional, Union

# Try to import optional dependencies
try:
    import logfire
    HAS_LOGFIRE = True
except ImportError:
    logfire = None
    HAS_LOGFIRE = False

class LogFireProvider:
    """Enhanced logging provider with direct HTTP logging to LogFire."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        service_name: str = "agent",
        environment: str = "development",
        project_id: Optional[str] = None,
        enable_console: bool = True
    ):
        """
        Initialize the enhanced LogFire logging provider.
        
        Args:
            api_key: LogFire API key (defaults to LOGFIRE_API_KEY env var)
            service_name: Name of the service for logging
            environment: Environment (development, staging, production)
            project_id: LogFire project ID (optional, for direct HTTP logging)
            enable_console: Whether to log to console as well
        """
        self.service_name = service_name
        self.environment = environment
        self.enable_console = enable_console
        
        # Get API key
        self.api_key = api_key or os.environ.get("LOGFIRE_API_KEY")
        self.project_id = project_id
        
        # Set up logging
        self._setup_logging()
        
        # Track conversation context
        self.conversation_id = None
        self.start_time = None
        self.messages = []
        
        # Initialize LogFire if available
        self.logfire_sdk_working = False
        if HAS_LOGFIRE and self.api_key:
            try:
                logfire.configure(token=self.api_key)
                print(f"✅ Initialized LogFire SDK with token")
                self.logfire_sdk_working = True
                
                # Test the connection
                try:
                    logfire.info("logfire_initialized", service=service_name, environment=environment)
                    print("✅ Successfully sent test log to LogFire")
                except Exception as e:
                    print(f"⚠️ LogFire SDK initialized but test log failed: {e}")
                    self.logfire_sdk_working = False
            except Exception as e:
                print(f"⚠️ Error initializing LogFire SDK: {e}")
                self.logfire_sdk_working = False
        else:
            if not HAS_LOGFIRE:
                print("⚠️ LogFire SDK not available, using HTTP fallback")
            if not self.api_key:
                print("⚠️ No LogFire API key provided, logs will only be saved locally")
        
        # Always attempt a direct HTTP log for verification
        if self.api_key:
            success = self._send_http_log("provider_initialized", {
                "service": service_name,
                "environment": environment,
                "sdk_available": HAS_LOGFIRE,
                "sdk_working": self.logfire_sdk_working
            })
            if success:
                print("✅ Successfully sent initialization log via HTTP")
            else:
                print("⚠️ Failed to send initialization log via HTTP")
    
    def _setup_logging(self):
        """Set up logging to console and file."""
        # Create logger
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        try:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # Regular log file
            file_handler = logging.FileHandler(os.path.join(log_dir, f"{self.service_name}.log"))
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # JSON log file
            self.json_log_file = os.path.join(log_dir, f"{self.service_name}.jsonl")
            
            print(f"📝 Logs will be saved to {os.path.join(log_dir, self.service_name)}.log and .jsonl")
        except Exception as e:
            print(f"⚠️ Error setting up file logging: {e}")
            self.json_log_file = None
    
    def _log_to_file(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event to the JSON log file."""
        if not self.json_log_file:
            return
            
        try:
            log_entry = {
                "timestamp": time.time(),
                "event": event_type,
                "service": self.service_name,
                "environment": self.environment,
                "conversation_id": self.conversation_id,
                **data
            }
            
            with open(self.json_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"⚠️ Error writing to JSON log file: {e}")
    
    def _send_http_log(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Send a log directly to LogFire via HTTP API."""
        if not self.api_key:
            return False
            
        try:
            import urllib.request
            import urllib.error
            
            # Prepare log data
            log_data = {
                "timestamp": time.time() * 1000,  # LogFire expects milliseconds
                "event": event_type,
                "service": self.service_name,
                "environment": self.environment,
                "level": "info",
                **data
            }
            
            if self.conversation_id:
                log_data["conversation_id"] = self.conversation_id
            
            # Convert any non-serializable values to strings
            for key, value in log_data.items():
                if isinstance(value, (dict, list, tuple)):
                    try:
                        json.dumps(value)  # Test if serializable
                    except (TypeError, OverflowError):
                        log_data[key] = str(value)
            
            # Determine endpoint URL
            endpoint = "https://api.logfire.dev/api/v1/log"
            if self.project_id:
                endpoint = f"https://api.logfire.dev/api/v1/projects/{self.project_id}/log"
            
            # Create and send request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(log_data).encode('utf-8'),
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200 or response.status == 201
                
        except Exception as e:
            print(f"⚠️ HTTP log error for event '{event_type}': {e}")
            return False
    
    def _log_to_logfire_sdk(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Log an event using the LogFire SDK."""
        if not HAS_LOGFIRE or not self.logfire_sdk_working:
            return False
            
        try:
            # Clone data to avoid modifying the original
            log_data = data.copy()
            
            # Add conversation_id if available
            if self.conversation_id and "conversation_id" not in log_data:
                log_data["conversation_id"] = self.conversation_id
            
            # Log the event
            logfire.info(event_type, **log_data)
            return True
        except Exception as e:
            print(f"⚠️ LogFire SDK error for event '{event_type}': {e}")
            self.logfire_sdk_working = False  # Mark as not working for future logs
            return False
    
    def _log_event(self, event_type: str, data: Dict[str, Any], log_message: Optional[str] = None) -> None:
        """Log an event through all available channels."""
        # Log to console/file logger
        if log_message:
            self.logger.info(log_message)
        
        # Log to JSON file
        self._log_to_file(event_type, data)
        
        # Try LogFire SDK first
        sdk_success = self._log_to_logfire_sdk(event_type, data)
        
        # Fall back to HTTP if SDK fails
        if not sdk_success and self.api_key:
            http_success = self._send_http_log(event_type, data)
            if not http_success:
                print(f"⚠️ Failed to send '{event_type}' log to LogFire")
    
    def start_conversation(self, user_id: Optional[str] = None) -> str:
        """Start a new conversation and return the conversation ID."""
        self.conversation_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.messages = []
        
        # Prepare event data
        event_data = {
            "user_id": user_id or "anonymous"
        }
        
        # Log the event
        self._log_event(
            "conversation_started", 
            event_data,
            f"Conversation started: {self.conversation_id}"
        )
        
        return self.conversation_id
    
    def log_user_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a user message."""
        if not self.conversation_id:
            self.start_conversation()
        
        # Store message
        msg_data = {
            "role": "user",
            "content": message,
            "timestamp": time.time()
        }
        if metadata:
            msg_data["metadata"] = metadata
        self.messages.append(msg_data)
        
        # Prepare event data
        event_data = {
            "message": message
        }
        if metadata:
            # Flatten metadata for easier querying
            for key, value in metadata.items():
                event_data[f"metadata_{key}"] = str(value) if isinstance(value, (dict, list, tuple)) else value
        
        # Log the event
        self._log_event(
            "user_message", 
            event_data,
            f"User: {message[:100]}..." if len(message) > 100 else f"User: {message}"
        )
    
    def log_agent_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an agent message."""
        if not self.conversation_id:
            self.start_conversation()
        
        # Store message
        msg_data = {
            "role": "agent",
            "content": message,
            "timestamp": time.time()
        }
        if metadata:
            msg_data["metadata"] = metadata
        self.messages.append(msg_data)
        
        # Prepare event data
        event_data = {
            "message": message[:500] + ("..." if len(message) > 500 else "")
        }
        if metadata:
            # Flatten metadata for easier querying
            for key, value in metadata.items():
                event_data[f"metadata_{key}"] = str(value) if isinstance(value, (dict, list, tuple)) else value
        
        # Log the event
        self._log_event(
            "agent_message", 
            event_data,
            f"Agent: {message[:100]}..." if len(message) > 100 else f"Agent: {message}"
        )
    
    def log_error(self, error: Union[str, Exception], context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error."""
        error_msg = str(error)
        error_type = type(error).__name__ if isinstance(error, Exception) else "Error"
        stack_trace = traceback.format_exc() if isinstance(error, Exception) else None
        
        # Prepare event data
        event_data = {
            "error_message": error_msg,
            "error_type": error_type
        }
        
        if stack_trace:
            event_data["stack_trace"] = stack_trace
            
        if context:
            # Flatten context for easier querying
            for key, value in context.items():
                event_data[f"context_{key}"] = str(value) if isinstance(value, (dict, list, tuple)) else value
        
        # Log the event
        self._log_event(
            "error", 
            event_data,
            f"Error: {error_msg}"
        )
    
    def log_system_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a system message."""
        # Prepare event data
        event_data = {
            "message": message
        }
        if metadata:
            # Flatten metadata for easier querying
            for key, value in metadata.items():
                event_data[f"metadata_{key}"] = str(value) if isinstance(value, (dict, list, tuple)) else value
        
        # Log the event
        self._log_event(
            "system_message", 
            event_data,
            f"System: {message}"
        )
    
    def end_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End the current conversation and log summary."""
        if not self.conversation_id:
            return
        
        # Calculate metrics
        duration = time.time() - self.start_time if self.start_time else 0
        user_messages = sum(1 for msg in self.messages if msg["role"] == "user")
        agent_messages = sum(1 for msg in self.messages if msg["role"] == "agent")
        
        # Prepare event data
        event_data = {
            "duration_seconds": duration,
            "user_message_count": user_messages,
            "agent_message_count": agent_messages,
            "total_message_count": len(self.messages)
        }
        
        if metadata:
            # Flatten metadata for easier querying
            for key, value in metadata.items():
                event_data[f"metadata_{key}"] = str(value) if isinstance(value, (dict, list, tuple)) else value
        
        # Log the event
        self._log_event(
            "conversation_ended", 
            event_data,
            f"Conversation ended: {self.conversation_id} (Duration: {duration:.2f}s)"
        )
        
        # Reset conversation state
        self.conversation_id = None
        self.start_time = None
        self.messages = []
    
    # Aliases for backward compatibility
    log_agent_response = log_agent_message