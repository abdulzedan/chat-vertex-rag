import React from 'react';
import { FileText } from 'lucide-react';
import { Document } from '@/types';

interface DocumentViewerProps {
  document: Document | null;
}

const DocumentViewer: React.FC<DocumentViewerProps> = ({ document }) => {
  if (!document) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center">
          <FileText className="h-12 w-12 mx-auto mb-4" />
          <p className="text-lg">Select a document to view</p>
        </div>
      </div>
    );
  }

  const isPDF = document.display_name.toLowerCase().endsWith('.pdf');
  const isImage = /\.(png|jpg|jpeg|gif)$/i.test(document.display_name);
  const isCSV = document.display_name.toLowerCase().endsWith('.csv');

  return (
    <div className="flex-1 flex flex-col">
      <div className="border-b p-4">
        <h1 className="text-xl font-semibold">{document.display_name}</h1>
        {document.description && (
          <p className="text-sm text-muted-foreground mt-1">{document.description}</p>
        )}
      </div>
      
      <div className="flex-1 p-8 flex items-center justify-center bg-muted/10">
        {isPDF && (
          <div className="text-center">
            <FileText className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
            <p className="text-lg font-medium mb-2">PDF Document</p>
            <p className="text-sm text-muted-foreground">
              PDF preview is available in the full version
            </p>
          </div>
        )}
        
        {isImage && (
          <div className="text-center">
            <p className="text-lg font-medium mb-2">Image Document</p>
            <p className="text-sm text-muted-foreground">
              Image preview is available in the full version
            </p>
          </div>
        )}
        
        {isCSV && (
          <div className="text-center">
            <p className="text-lg font-medium mb-2">CSV Data</p>
            <p className="text-sm text-muted-foreground">
              Data preview is available in the full version
            </p>
          </div>
        )}
        
        {!isPDF && !isImage && !isCSV && (
          <div className="text-center">
            <p className="text-lg font-medium mb-2">Document</p>
            <p className="text-sm text-muted-foreground">
              Document content will be processed by the RAG engine
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentViewer;