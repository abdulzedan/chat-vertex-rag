export interface Document {
  id: string;
  display_name: string;
  description?: string;
  create_time: string;
  update_time: string;
  size_bytes?: number;
}

export interface UploadResponse {
  id: string;
  display_name: string;
  file_type: string;
  upload_time: string;
  status: string;
}

export interface Citation {
  filename: string;
  document_id: string;
  relevance_score: number;
  pages?: string[];
  sections?: string[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  citations?: Citation[];
}

export interface ActivityEvent {
  type: 'activity' | 'activity_clear';
  timestamp: string;
  stage?: string;
  message?: string;
  detail?: string;
  event_type?: 'info' | 'success' | 'warning' | 'error';
}
