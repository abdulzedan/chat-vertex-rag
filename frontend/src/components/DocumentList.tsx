import React, { useState, useEffect } from 'react';
import { FileText, Image, Table, Trash2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { Document } from '@/types';
import FileUploader from './FileUploader';

interface DocumentListProps {
  selectedDocument: Document | null;
  onSelectDocument: (doc: Document) => void;
}

const DocumentList: React.FC<DocumentListProps> = ({ selectedDocument, onSelectDocument }) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUploader, setShowUploader] = useState(false);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/documents');
      if (response.ok) {
        const data = await response.json();
        setDocuments(data);
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleDelete = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const response = await fetch(`/api/documents/${docId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        fetchDocuments();
      }
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };

  const getFileIcon = (displayName: string) => {
    const ext = displayName.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':
        return <FileText className="h-4 w-4" />;
      case 'png':
      case 'jpg':
      case 'jpeg':
        return <Image className="h-4 w-4" />;
      case 'csv':
        return <Table className="h-4 w-4" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Documents</h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={fetchDocuments}
            className="h-8 w-8"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        
        {showUploader ? (
          <FileUploader 
            onUploadComplete={() => {
              fetchDocuments();
              setShowUploader(false);
            }}
            onCancel={() => setShowUploader(false)}
          />
        ) : (
          <Button 
            onClick={() => setShowUploader(true)}
            className="w-full"
            variant="outline"
          >
            Upload Document
          </Button>
        )}
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 pt-0 space-y-2">
          {loading ? (
            <>
              {[...Array(3)].map((_, i) => (
                <div key={i} className="p-3 rounded-lg border">
                  <Skeleton className="h-4 w-3/4 mb-2" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              ))}
            </>
          ) : documents.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No documents uploaded yet
            </div>
          ) : (
            documents.map((doc) => (
              <div
                key={doc.id}
                className={`p-3 rounded-lg border cursor-pointer transition-colors hover:bg-accent ${
                  selectedDocument?.id === doc.id ? 'bg-accent' : ''
                }`}
                onClick={() => onSelectDocument(doc)}
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5">
                    {getFileIcon(doc.display_name)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm truncate">
                      {doc.display_name}
                    </p>
                    {doc.description && (
                      <p className="text-xs text-muted-foreground truncate">
                        {doc.description}
                      </p>
                    )}
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatDate(doc.create_time)}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 opacity-0 hover:opacity-100"
                    onClick={(e) => handleDelete(doc.id, e)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
};

export default DocumentList;