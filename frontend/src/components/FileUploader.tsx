import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Upload, FileText, Image, Table, X, Loader2, Link } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FileUploaderProps {
  onUploadComplete: () => void;
  onCancel: () => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onUploadComplete, onCancel }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [driveUrl, setDriveUrl] = useState('');
  const [uploadMode, setUploadMode] = useState<'drive' | 'file'>('drive');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setUploading(true);
    setUploadError(null);

    try {
      for (const file of acceptedFiles) {
        const formData = new FormData();
        formData.append('file', file);

        // Create an AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
        
        const response = await fetch('/api/documents/upload', {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        });
        
        clearTimeout(timeoutId);

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Upload failed');
        }
      }

      onUploadComplete();
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          setUploadError('Upload timed out. Large files may still be processing in the background.');
        } else {
          setUploadError(error.message);
        }
      } else {
        setUploadError('Upload failed');
      }
    } finally {
      setUploading(false);
    }
  }, [onUploadComplete]);

  const handleDriveUrl = useCallback(async () => {
    if (!driveUrl.trim()) return;

    // Validate Google Drive URL
    const driveUrlPattern = /^https:\/\/drive\.google\.com\/(file\/d\/[a-zA-Z0-9_-]+|drive\/folders\/[a-zA-Z0-9_-]+)/;
    if (!driveUrlPattern.test(driveUrl)) {
      setUploadError('Please enter a valid Google Drive URL (file or folder)');
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      const response = await fetch('/api/documents/import-drive', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ drive_url: driveUrl }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Import failed');
      }

      onUploadComplete();
      setDriveUrl('');
    } catch (error) {
      if (error instanceof Error) {
        setUploadError(error.message);
      } else {
        setUploadError('Import failed');
      }
    } finally {
      setUploading(false);
    }
  }, [driveUrl, onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg'],
      'text/csv': ['.csv'],
    },
    onDrop,
    disabled: uploading,
  });

  return (
    <div className="space-y-3">
      {/* Mode selector */}
      <div className="flex gap-2">
        <Button
          variant={uploadMode === 'drive' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setUploadMode('drive')}
          className="flex-1"
        >
          <Link className="h-4 w-4 mr-2" />
          Google Drive
        </Button>
        <Button
          variant={uploadMode === 'file' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setUploadMode('file')}
          className="flex-1"
        >
          <Upload className="h-4 w-4 mr-2" />
          Upload File
        </Button>
      </div>

      {/* Google Drive mode */}
      {uploadMode === 'drive' && (
        <Card className="p-4">
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Link className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Google Drive URL</span>
            </div>
            
            <div className="space-y-2">
              <Input
                placeholder="https://drive.google.com/file/d/... or https://drive.google.com/drive/folders/..."
                value={driveUrl}
                onChange={(e) => setDriveUrl(e.target.value)}
                disabled={uploading}
              />
              <p className="text-xs text-muted-foreground">
                Share the file with: <code className="bg-muted px-1 rounded text-xs">service-{'{project_number}'}@gcp-sa-vertex-rag.iam.gserviceaccount.com</code>
              </p>
            </div>
            
            <Button 
              onClick={handleDriveUrl}
              disabled={uploading || !driveUrl.trim()}
              className="w-full"
            >
              {uploading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Importing...
                </>
              ) : (
                <>
                  <Link className="h-4 w-4 mr-2" />
                  Import from Drive
                </>
              )}
            </Button>
          </div>
        </Card>
      )}

      {/* File upload mode */}
      {uploadMode === 'file' && (
        <Card
          {...getRootProps()}
          className={cn(
            'border-2 border-dashed p-6 text-center cursor-pointer transition-colors relative',
            isDragActive && 'border-primary bg-primary/5',
            uploading && 'opacity-50 cursor-not-allowed'
          )}
        >
          <input {...getInputProps()} />
          
          {uploading ? (
            <div>
              <Loader2 className="h-8 w-8 mx-auto mb-3 animate-spin text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Uploading...</p>
            </div>
          ) : (
            <>
              <Upload className="h-8 w-8 mx-auto mb-3 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                {isDragActive
                  ? 'Drop files here'
                  : 'Drop files here or click to browse'}
              </p>
              <div className="flex gap-2 justify-center mt-2">
                <FileText className="h-4 w-4 text-muted-foreground" />
                <Image className="h-4 w-4 text-muted-foreground" />
                <Table className="h-4 w-4 text-muted-foreground" />
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                PDF, Images (PNG, JPG), CSV
              </p>
            </>
          )}
        </Card>
      )}

      {uploadError && (
        <div className="text-sm text-destructive text-center">
          {uploadError}
        </div>
      )}

      {!uploading && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onCancel}
          className="w-full"
        >
          <X className="h-4 w-4 mr-2" />
          Cancel
        </Button>
      )}
    </div>
  );
};

export default FileUploader;