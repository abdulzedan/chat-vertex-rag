import { useState } from 'react';
import DocumentList from '@/components/DocumentList';
import DocumentViewer from '@/components/DocumentViewer';
import ChatInterface from '@/components/ChatInterface';
import { Document } from '@/types';

function App() {
  // State for multi-document selection
  const [selectedDocuments, setSelectedDocuments] = useState<Document[]>([]);
  // Keep single document selection for viewer
  const [viewerDocument, setViewerDocument] = useState<Document | null>(null);

  // Handle document selection for chat/search
  const handleDocumentSelect = (document: Document, isSelected: boolean) => {
    if (isSelected) {
      setSelectedDocuments(prev => [...prev, document]);
    } else {
      setSelectedDocuments(prev => prev.filter(doc => doc.id !== document.id));
    }
  };

  // Handle selecting all/none documents
  const handleSelectAll = (documents: Document[], selectAll: boolean) => {
    if (selectAll) {
      setSelectedDocuments(documents);
    } else {
      setSelectedDocuments([]);
    }
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Document List - 25% width */}
      <div className="w-1/4 min-w-[300px] border-r">
        <DocumentList 
          selectedDocuments={selectedDocuments}
          viewerDocument={viewerDocument}
          onSelectDocument={handleDocumentSelect}
          onSelectAll={handleSelectAll}
          onViewDocument={setViewerDocument}
        />
      </div>
      
      {/* Document Viewer - Remaining space */}
      <div className="flex-1 flex flex-col">
        <DocumentViewer document={viewerDocument} />
      </div>
      
      {/* Chat Interface - Floating button + slide-out sheet */}
      <ChatInterface selectedDocuments={selectedDocuments} />
    </div>
  );
}

export default App;