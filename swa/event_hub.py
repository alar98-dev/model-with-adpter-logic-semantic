from typing import Dict, List, Any, Callable, Optional
from threading import Lock
from swa.depth_analyzer import RuntimeEvent
import time


class SWAEventHub:
    """
    Pub/Sub event hub for SWA Broadcast Eventual Runtime.
    
    Manages event subscriptions and broadcasts events to registered listeners.
    Supports filtering by event type and asynchronous delivery.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._hub_lock = Lock()
        self._event_history: List[RuntimeEvent] = []
        self._max_history = 100  # Keep last 100 events

    def subscribe(self, event_type: str, callback: Callable[[RuntimeEvent], None]) -> None:
        """Subscribe to events of a specific type."""
        with self._hub_lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[RuntimeEvent], None]) -> None:
        """Unsubscribe from events of a specific type."""
        with self._hub_lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    # Clean up empty lists
                    if not self._subscribers[event_type]:
                        del self._subscribers[event_type]
                except ValueError:
                    pass  # Callback not found

    def broadcast(self, event: RuntimeEvent) -> None:
        """Broadcast event to all subscribers of its type."""
        # Store in history first
        self._add_to_history(event)
        
        # Get subscribers for this event type
        subscribers = self._get_subscribers(event.type)
        
        # Broadcast to each subscriber asynchronously
        for callback in subscribers:
            try:
                # In a real implementation, you might want to use threading or async
                callback(event)
            except Exception as e:
                print(f"Event delivery failed for {event.type}: {e}")

    def broadcast_all(self, event: RuntimeEvent) -> None:
        """Broadcast event to ALL subscribers regardless of type."""
        self._add_to_history(event)
        
        all_subscribers = []
        with self._hub_lock:
            for subscribers in self._subscribers.values():
                all_subscribers.extend(subscribers)
        
        for callback in all_subscribers:
            try:
                callback(event)
            except Exception as e:
                print(f"Event delivery failed: {e}")

    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 10) -> List[RuntimeEvent]:
        """Get recent events, optionally filtered by type."""
        with self._hub_lock:
            if event_type:
                filtered = [e for e in self._event_history if e.type == event_type]
                return filtered[-limit:]
            else:
                return self._event_history[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        with self._hub_lock:
            self._event_history.clear()

    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """Get number of subscribers for an event type or total."""
        with self._hub_lock:
            if event_type:
                return len(self._subscribers.get(event_type, []))
            else:
                return sum(len(subs) for subs in self._subscribers.values())

    def _get_subscribers(self, event_type: str) -> List[Callable]:
        """Get list of subscribers for an event type (thread-safe)."""
        with self._hub_lock:
            return self._subscribers.get(event_type, []).copy()

    def _add_to_history(self, event: RuntimeEvent) -> None:
        """Add event to history (thread-safe)."""
        with self._hub_lock:
            self._event_history.append(event)
            # Maintain max history size
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)


# Convenience functions for common event types
def create_vision_obstacle_event(direction: str, confidence: float, depth: float = 1.0) -> RuntimeEvent:
    """Create a vision obstacle event."""
    return RuntimeEvent(
        type="vision_obstacle",
        payload={
            "direction": direction,
            "confidence": confidence,
            "depth": depth,
            "timestamp": time.time()
        }
    )

def create_user_steering_event(command: str, strength: float = 1.0) -> RuntimeEvent:
    """Create a user steering event."""
    return RuntimeEvent(
        type="user_steering",
        payload={
            "command": command,
            "strength": strength,
            "timestamp": time.time()
        }
    )

def create_state_change_event(component: str, old_state: Any, new_state: Any) -> RuntimeEvent:
    """Create a state change event."""
    return RuntimeEvent(
        type="state_change",
        payload={
            "component": component,
            "old_state": old_state,
            "new_state": new_state,
            "timestamp": time.time()
        }
    )