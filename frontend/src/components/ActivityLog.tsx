import { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, CheckCircle2, AlertTriangle, XCircle, Loader2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { ActivityEvent } from '@/types';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const WS_BASE = API_BASE.replace(/^http/, 'ws');

function getEventIcon(eventType?: string) {
  switch (eventType) {
    case 'success':
      return <CheckCircle2 className='h-4 w-4 text-green-500' />;
    case 'warning':
      return <AlertTriangle className='h-4 w-4 text-yellow-500' />;
    case 'error':
      return <XCircle className='h-4 w-4 text-red-500' />;
    case 'info':
    default:
      return <Loader2 className='h-4 w-4 animate-spin text-blue-500' />;
  }
}

function formatTime(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '';
  }
}

export default function ActivityLog() {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const mountedRef = useRef(true);

  const handleEvent = useCallback((data: ActivityEvent) => {
    if (data.type === 'activity_clear') {
      setEvents([]);
    } else if (data.type === 'activity') {
      setEvents((prev) => {
        // When a completion event (success/warning/error) arrives for a stage,
        // replace any existing "info" (spinner) event for that same stage.
        if (data.event_type !== 'info' && data.stage) {
          const filtered = prev.filter(
            (e) => !(e.stage === data.stage && e.event_type === 'info')
          );
          return [...filtered, data];
        }
        return [...prev, data];
      });
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    function connect() {
      // Don't reconnect if unmounted
      if (!mountedRef.current) return;

      const ws = new WebSocket(`${WS_BASE}/ws/activity`);
      wsRef.current = ws;

      ws.onopen = () => {
        if (mountedRef.current) setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data: ActivityEvent = JSON.parse(event.data);
          handleEvent(data);
        } catch {
          // ignore
        }
      };

      ws.onclose = () => {
        if (mountedRef.current) {
          setConnected(false);
          // Only reconnect if still mounted
          reconnectTimer.current = setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect on cleanup close
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [handleEvent]);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  // Keep-alive ping
  useEffect(() => {
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className='flex h-full flex-col'>
      {/* Header */}
      <div className='flex items-center justify-between border-b px-6 py-4'>
        <div className='flex items-center gap-3'>
          <Activity className='h-5 w-5 text-muted-foreground' />
          <h2 className='text-lg font-semibold'>Activity Log</h2>
          {events.length > 0 && (
            <Badge variant='secondary' className='text-xs'>
              {events.length} events
            </Badge>
          )}
        </div>
        <div className='flex items-center gap-2 text-xs text-muted-foreground'>
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              connected ? 'bg-green-500' : 'bg-red-400'
            }`}
          />
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      {/* Event list */}
      <div ref={scrollRef} className='flex-1 overflow-y-auto px-6 py-4'>
        {events.length === 0 ? (
          <div className='flex h-full items-center justify-center'>
            <div className='text-center'>
              <Activity className='mx-auto mb-3 h-10 w-10 text-muted-foreground/30' />
              <p className='text-sm text-muted-foreground/60'>
                Pipeline activity will appear here during uploads and queries.
              </p>
            </div>
          </div>
        ) : (
          <div className='space-y-2'>
            {events.map((event, idx) => (
              <div
                key={idx}
                className='flex items-start gap-3 rounded-lg border border-transparent px-3 py-2 transition-colors hover:border-border hover:bg-muted/40'
              >
                <div className='mt-0.5 flex-shrink-0'>
                  {getEventIcon(event.event_type)}
                </div>
                <div className='min-w-0 flex-1'>
                  <div className='flex items-baseline gap-2'>
                    <span className='text-sm font-medium text-foreground'>
                      {event.message}
                    </span>
                    {event.stage && (
                      <Badge variant='outline' className='text-[10px] px-1.5 py-0'>
                        {event.stage}
                      </Badge>
                    )}
                    <span className='ml-auto flex-shrink-0 text-xs text-muted-foreground/50'>
                      {formatTime(event.timestamp || '')}
                    </span>
                  </div>
                  {event.detail && (
                    <pre className='mt-1 whitespace-pre-wrap font-sans text-xs text-muted-foreground'>
                      {event.detail}
                    </pre>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
