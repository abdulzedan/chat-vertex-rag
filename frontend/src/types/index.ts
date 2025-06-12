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

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}