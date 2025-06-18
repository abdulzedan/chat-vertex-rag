import React, { useState, useEffect } from 'react';
import { FileText, Image, Table, Trash2, RefreshCw, Eye, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { Checkbox } from '@/components/ui/checkbox';
import { 
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { Document } from '@/types';
import FileUploader from './FileUploader';

interface DocumentListProps {
  selectedDocuments: Document[];
  viewerDocument: Document | null;
  onSelectDocument: (doc: Document, isSelected: boolean) => void;
  onSelectAll: (documents: Document[], selectAll: boolean) => void;
  onViewDocument: (doc: Document) => void;
}

const DocumentList: React.FC<DocumentListProps> = ({ 
  selectedDocuments, 
  viewerDocument, 
  onSelectDocument, 
  onSelectAll, 
  onViewDocument 
}) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUploader, setShowUploader] = useState(false);
  const [clearingAll, setClearingAll] = useState(false);

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

  const handleDelete = async (docId: string, docName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    // Show confirmation dialog
    const confirmed = window.confirm(`Are you sure you want to delete "${docName}"?`);
    if (!confirmed) return;
    
    try {
      const response = await fetch(`/api/documents/${docId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        // Remove from selected documents if it was selected
        onSelectDocument(documents.find(d => d.id === docId)!, false);
        fetchDocuments();
      } else {
        console.error('Failed to delete document:', await response.text());
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

  // Check if a document is selected for search
  const isDocumentSelected = (doc: Document) => {
    return selectedDocuments.some(selected => selected.id === doc.id);
  };

  // Check if all documents are selected
  const areAllSelected = documents.length > 0 && selectedDocuments.length === documents.length;
  const areSomeSelected = selectedDocuments.length > 0 && selectedDocuments.length < documents.length;

  // Handle select all toggle
  const handleSelectAllToggle = () => {
    onSelectAll(documents, !areAllSelected);
  };

  const handleClearAllDocuments = async () => {
    setClearingAll(true);
    try {
      const response = await fetch('/api/documents/', {
        method: 'DELETE',
      });
      
      if (response.ok) {
        // Clear local state
        setDocuments([]);
        // Clear parent state
        onSelectAll([], false);
        // Refresh to make sure
        await fetchDocuments();
      } else {
        console.error('Failed to clear all documents');
      }
    } catch (error) {
      console.error('Error clearing all documents:', error);
    } finally {
      setClearingAll(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold">Documents</h2>
            {documents.length > 0 && (
              <span className="text-sm text-muted-foreground">
                ({selectedDocuments.length}/{documents.length} selected)
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            {documents.length > 0 && (
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs text-destructive hover:text-destructive hover:bg-destructive/10 border-destructive/20"
                    title="Clear all documents from the system"
                  >
                    <Trash2 className="h-3 w-3 mr-1" />
                    Clear All
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle className="flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5 text-destructive" />
                      Clear All Documents
                    </AlertDialogTitle>
                    <AlertDialogDescription>
                      This will permanently delete all {documents.length} documents from the system, including their content and search index. This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      onClick={handleClearAllDocuments}
                      className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                      disabled={clearingAll}
                    >
                      {clearingAll ? 'Clearing...' : 'Clear All Documents'}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={fetchDocuments}
              className="h-8 w-8"
              title="Refresh documents"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {documents.length > 0 && (
          <div className="flex items-center gap-2">
            <Checkbox
              checked={areAllSelected}
              indeterminate={areSomeSelected}
              onCheckedChange={handleSelectAllToggle}
              id="select-all"
            />
            <label htmlFor="select-all" className="text-sm cursor-pointer">
              {areAllSelected ? 'Deselect all' : areSomeSelected ? 'Select all' : 'Select all'}
            </label>
            {selectedDocuments.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onSelectAll(documents, false)}
                className="h-6 px-2 text-xs"
              >
                Clear
              </Button>
            )}
          </div>
        )}
        
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
                className={`p-3 rounded-lg border transition-colors ${
                  viewerDocument?.id === doc.id ? 'bg-accent border-primary' : 'hover:bg-accent/50'
                } ${
                  isDocumentSelected(doc) ? 'bg-primary/5 border-primary/20' : ''
                }`}
              >
                <div className="flex items-start gap-3">
                  {/* Selection checkbox */}
                  <div className="mt-0.5">
                    <Checkbox
                      checked={isDocumentSelected(doc)}
                      onCheckedChange={(checked: boolean) => onSelectDocument(doc, checked)}
                      id={`doc-${doc.id}`}
                    />
                  </div>
                  
                  {/* File icon */}
                  <div className="mt-0.5">
                    {getFileIcon(doc.display_name)}
                  </div>
                  
                  {/* Document info - clickable for viewing */}
                  <div className="flex-1 min-w-0 cursor-pointer" onClick={() => onViewDocument(doc)}>
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
                  
                  {/* Action buttons */}
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 opacity-60 hover:opacity-100"
                      onClick={() => onViewDocument(doc)}
                      title="View document"
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 opacity-60 hover:opacity-100 text-destructive hover:text-destructive"
                      onClick={(e) => handleDelete(doc.id, doc.display_name, e)}
                      title="Delete this document"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
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