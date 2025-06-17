import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, Loader2, Trash2, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';
import ReactMarkdown from 'react-markdown';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { ChatMessage, Document } from '@/types';

interface ChatInterfaceProps {
  selectedDocuments: Document[];
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ selectedDocuments }) => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(() => {
    const id = crypto.randomUUID();
    console.log('Generated new session ID:', id);
    return id;
  });
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const requestBody = {
        question: input,
        similarity_top_k: selectedDocuments.length > 0 ? selectedDocuments.length * 25 : 10,
        vector_distance_threshold: 0.5,
        document_ids: selectedDocuments.length > 0 ? selectedDocuments.map(doc => doc.id) : null,
        session_id: sessionId,
      };
      
      const response = await fetch('/api/chat/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) throw new Error('Failed to send message');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      let assistantMessage: ChatMessage = {
        role: 'assistant',
        content: '',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.content) {
                  assistantMessage.content += data.content;
                  setMessages((prev) => {
                    const newMessages = [...prev];
                    newMessages[newMessages.length - 1] = { ...assistantMessage };
                    return newMessages;
                  });
                }
              } catch (e) {
                // Ignore parse errors
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(crypto.randomUUID()); // Generate new session ID when clearing chat
  };

  return (
    <>
      <div className="fixed bottom-4 right-4">
        <Button
          onClick={() => setOpen(true)}
          className="relative rounded-full h-12 w-12 shadow-lg transition-all duration-200 hover:scale-105"
          size="icon"
        >
          <MessageSquare className="h-5 w-5" />
          {selectedDocuments.length > 0 && (
            <Badge 
              variant="destructive" 
              className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center p-0 text-xs"
            >
              {selectedDocuments.length}
            </Badge>
          )}
        </Button>
      </div>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent className="w-[400px] sm:w-[540px] flex flex-col p-0">
          <SheetHeader className="px-6 py-4 border-b space-y-3">
            <div className="flex items-center justify-between">
              <SheetTitle>Document Q&A</SheetTitle>
              {messages.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearChat}
                  className="h-8 px-3 text-xs"
                >
                  <Trash2 className="h-3 w-3 mr-1" />
                  Clear Chat
                </Button>
              )}
            </div>
            
            {/* Document selection info */}
            <div className="flex flex-col gap-2">
              {selectedDocuments.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  <span className="text-xs text-muted-foreground">Searching in:</span>
                  {selectedDocuments.slice(0, 3).map((doc) => (
                    <Badge key={doc.id} variant="secondary" className="text-xs">
                      <FileText className="h-3 w-3 mr-1" />
                      {doc.display_name.length > 15 
                        ? `${doc.display_name.slice(0, 15)}...` 
                        : doc.display_name}
                    </Badge>
                  ))}
                  {selectedDocuments.length > 3 && (
                    <Badge variant="secondary" className="text-xs">
                      +{selectedDocuments.length - 3} more
                    </Badge>
                  )}
                </div>
              ) : (
                <div className="text-xs text-muted-foreground flex items-center gap-1">
                  <FileText className="h-3 w-3" />
                  Searching all documents
                </div>
              )}
            </div>
          </SheetHeader>
          
          <ScrollArea className="flex-1 p-6" ref={scrollAreaRef}>
            <div className="space-y-4">
              {messages.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  <MessageSquare className="h-10 w-10 mx-auto mb-3" />
                  <p>Ask questions about your documents</p>
                </div>
              ) : (
                messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    } animate-in fade-in-50 slide-in-from-bottom-2 duration-300`}
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-2 transition-all duration-200 ${
                        message.role === 'user'
                          ? 'bg-primary text-primary-foreground shadow-sm'
                          : 'bg-muted shadow-sm hover:shadow-md'
                      }`}
                    >
                      {message.content ? (
                        message.role === 'assistant' ? (
                          <ReactMarkdown className="prose prose-sm max-w-none">
                            {message.content}
                          </ReactMarkdown>
                        ) : (
                          <div>{message.content}</div>
                        )
                      ) : (
                        <div className="space-y-2">
                          <Skeleton className="h-4 w-[200px]" />
                          <Skeleton className="h-4 w-[160px]" />
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              {isLoading && messages[messages.length - 1]?.role === 'assistant' && !messages[messages.length - 1]?.content && (
                <div className="flex justify-start animate-in fade-in-50 slide-in-from-bottom-2 duration-300">
                  <div className="bg-muted rounded-lg px-4 py-2 shadow-sm">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm text-muted-foreground">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
          
          <div className="border-t p-4 bg-background/95 backdrop-blur-sm">
            <div className="flex gap-2">
              <Input
                placeholder={
                  selectedDocuments.length > 0 
                    ? `Ask about ${selectedDocuments.length} selected document${selectedDocuments.length > 1 ? 's' : ''}...`
                    : "Ask about your documents..."
                }
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                disabled={isLoading}
                className="transition-all duration-200 focus:shadow-sm"
              />
              <Button
                onClick={handleSendMessage}
                disabled={!input.trim() || isLoading}
                size="icon"
                className="transition-all duration-200 hover:scale-105 disabled:hover:scale-100"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
};

export default ChatInterface;