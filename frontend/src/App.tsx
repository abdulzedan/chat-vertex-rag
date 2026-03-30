import { useState } from 'react';
import DocumentList from '@/components/DocumentList';
import ChatInterface from '@/components/ChatInterface';
import ActivityLog from '@/components/ActivityLog';
import { Document } from '@/types';

function App() {
  const [selectedDocuments, setSelectedDocuments] = useState<Document[]>([]);

  const handleDocumentSelect = (document: Document, isSelected: boolean) => {
    if (isSelected) {
      setSelectedDocuments(prev => [...prev, document]);
    } else {
      setSelectedDocuments(prev => prev.filter(doc => doc.id !== document.id));
    }
  };

  const handleSelectAll = (documents: Document[], selectAll: boolean) => {
    if (selectAll) {
      setSelectedDocuments(documents);
    } else {
      setSelectedDocuments([]);
    }
  };

  return (
    <div className='flex h-screen bg-background'>
      {/* Document List */}
      <div className='w-1/4 min-w-[280px] border-r'>
        <DocumentList
          selectedDocuments={selectedDocuments}
          onSelectDocument={handleDocumentSelect}
          onSelectAll={handleSelectAll}
        />
      </div>

      {/* Chat — always visible */}
      <div className='flex flex-1 flex-col border-r'>
        <ChatInterface selectedDocuments={selectedDocuments} />
      </div>

      {/* Activity Log */}
      <div className='flex w-1/4 min-w-[300px] flex-col'>
        <ActivityLog />
      </div>
    </div>
  );
}

export default App;
