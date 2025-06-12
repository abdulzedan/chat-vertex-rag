import React, { useState } from 'react';
import { cn } from '@/lib/utils';
import { Separator } from '@/components/ui/separator';
import DocumentList from '@/components/DocumentList';
import DocumentViewer from '@/components/DocumentViewer';
import ChatInterface from '@/components/ChatInterface';
import { Document } from '@/types';

function App() {
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);

  return (
    <div className="flex h-screen bg-background">
      {/* Document List - 25% width */}
      <div className="w-1/4 min-w-[300px] border-r">
        <DocumentList 
          selectedDocument={selectedDocument}
          onSelectDocument={setSelectedDocument}
        />
      </div>
      
      {/* Document Viewer - Remaining space */}
      <div className="flex-1 flex flex-col">
        <DocumentViewer document={selectedDocument} />
      </div>
      
      {/* Chat Interface - Floating button + slide-out sheet */}
      <ChatInterface />
    </div>
  );
}

export default App;