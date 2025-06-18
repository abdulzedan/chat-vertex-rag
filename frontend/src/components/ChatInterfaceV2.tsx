import React, { useState, useRef, useEffect } from 'react';
import { flushSync } from 'react-dom';
import { Send, Loader2, Trash2, FileText, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { ChatMessage, Document } from '@/types';

interface ChatInterfaceProps {
  selectedDocuments: Document[];
}

interface Citation {
  id: number;
  text: string;
  source: string;
}

// Thinking animation component
const ThinkingDots: React.FC = () => {
  return (
    <span className="inline-flex">
      <span className="animate-bounce" style={{ animationDelay: '0ms' }}>.</span>
      <span className="animate-bounce" style={{ animationDelay: '150ms' }}>.</span>
      <span className="animate-bounce" style={{ animationDelay: '300ms' }}>.</span>
    </span>
  );
};

// Helper function to format content with citations
const formatMessageContent = (content: string): React.ReactNode => {
  const citations: Citation[] = [];
  let citationCounter = 1;
  let processedContent = content;

  // Extract and process citations from the content
  // Look for patterns like "according to Document X" or "Document X states" or "(Document X)"
  const citationPatterns = [
    /according to (Document \d+: [^,\.\n]+)/gi,
    /(Document \d+: [^,\.\n]+) (?:states|mentions|shows|indicates|reports)/gi,
    /\((Document \d+: [^\)]+)\)/gi,
    /(?:Document\s+\d+:\s*|---\s*Document\s+\d+:\s*)([^\n]+?)(?:\s*---)?/gi
  ];

  citationPatterns.forEach(pattern => {
    processedContent = processedContent.replace(pattern, (match, docRef) => {
      // Extract the document name
      const docMatch = docRef.match(/Document \d+: (.+)/) || [null, docRef];
      const docName = (docMatch[1] || docRef).trim();
      
      // Check if we already have this citation
      let existingCitation = citations.find(c => c.source === docName);
      if (!existingCitation) {
        existingCitation = {
          id: citationCounter++,
          text: match,
          source: docName
        };
        citations.push(existingCitation);
      }
      
      // Replace with superscript citation
      return `[${existingCitation.id}]`;
    });
  });

  // Also handle explicit references section if present
  const parts = processedContent.split(/\n(?=References:|Sources:|Citations:)/i);
  if (parts.length > 1) {
    processedContent = parts[0];
    // Parse references from the references section
    const refLines = parts[1].split('\n').slice(1); // Skip the "References:" line
    refLines.forEach(line => {
      const match = line.match(/^(?:\d+\.?\s*)?(.+)$/); 
      if (match && match[1].trim()) {
        const source = match[1].trim();
        if (!citations.find(c => c.source === source)) {
          citations.push({
            id: citationCounter++,
            text: '',
            source: source
          });
        }
      }
    });
  }

  return (
    <>
      <FormattedMessage content={processedContent} citations={citations} />
      {citations.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border">
          <p className="text-xs font-semibold text-muted-foreground mb-2">References:</p>
          <div className="text-xs text-muted-foreground space-y-1">
            {citations.map((citation) => (
              <div key={citation.id} className="flex gap-2">
                <span className="text-primary">[{citation.id}]</span>
                <span>{citation.source}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
};

// Component to format message content with proper styling
const FormattedMessage: React.FC<{ content: string; citations?: Citation[] }> = ({ content }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Parse content for special formatting
  const formatText = (text: string): React.ReactNode => {
    // Handle tables
    if (text.includes('|') && text.split('\n').some(line => line.includes('|'))) {
      return formatTable(text);
    }

    // Handle lists
    const lines = text.split('\n');
    const elements: React.ReactNode[] = [];
    let currentList: string[] = [];
    let listType: 'ul' | 'ol' | null = null;

    lines.forEach((line, idx) => {
      // Check for bullet points
      if (line.match(/^[\*\-•]\s+/)) {
        if (listType !== 'ul') {
          if (currentList.length > 0) {
            elements.push(createList(currentList, listType!));
            currentList = [];
          }
          listType = 'ul';
        }
        currentList.push(line.replace(/^[\*\-•]\s+/, ''));
      }
      // Check for numbered lists
      else if (line.match(/^\d+\.\s+/)) {
        if (listType !== 'ol') {
          if (currentList.length > 0) {
            elements.push(createList(currentList, listType!));
            currentList = [];
          }
          listType = 'ol';
        }
        currentList.push(line.replace(/^\d+\.\s+/, ''));
      }
      // Regular paragraph
      else {
        if (currentList.length > 0) {
          elements.push(createList(currentList, listType!));
          currentList = [];
          listType = null;
        }
        if (line.trim()) {
          elements.push(
            <p key={idx} className="text-sm leading-relaxed mb-3">
              {formatInlineElements(line)}
            </p>
          );
        }
      }
    });

    // Handle remaining list items
    if (currentList.length > 0) {
      elements.push(createList(currentList, listType!));
    }

    return <>{elements}</>;
  };

  // Format inline elements (bold, italic, code, citations)
  const formatInlineElements = (text: string): React.ReactNode => {
    // Replace citations [1], [2] etc with superscript
    let formatted = text.split(/(\[\d+\])/g).map((part, idx) => {
      if (part.match(/^\[\d+\]$/)) {
        const num = part.match(/\d+/)![0];
        return (
          <sup key={idx} className="text-xs text-primary cursor-help ml-0.5" title={`Reference ${num}`}>
            [{num}]
          </sup>
        );
      }
      
      // Handle bold text
      return part.split(/(\*\*[^*]+\*\*)/g).map((subPart, subIdx) => {
        if (subPart.match(/^\*\*[^*]+\*\*$/)) {
          return <strong key={`${idx}-${subIdx}`} className="font-semibold">{subPart.slice(2, -2)}</strong>;
        }
        
        // Handle inline code
        return subPart.split(/(`[^`]+`)/g).map((codePart, codeIdx) => {
          if (codePart.match(/^`[^`]+`$/)) {
            return (
              <code key={`${idx}-${subIdx}-${codeIdx}`} className="text-xs bg-muted px-1 py-0.5 rounded">
                {codePart.slice(1, -1)}
              </code>
            );
          }
          return codePart;
        });
      });
    });

    return <>{formatted}</>;
  };

  // Create a formatted list
  const createList = (items: string[], type: 'ul' | 'ol'): React.ReactNode => {
    const ListComponent = type === 'ul' ? 'ul' : 'ol';
    const className = type === 'ul' ? 'list-disc pl-5 space-y-1 mb-3' : 'list-decimal pl-5 space-y-1 mb-3';
    
    return (
      <ListComponent key={Math.random()} className={className}>
        {items.map((item, idx) => (
          <li key={idx} className="text-sm">
            {formatInlineElements(item)}
          </li>
        ))}
      </ListComponent>
    );
  };

  // Format tables
  const formatTable = (text: string): React.ReactNode => {
    const lines = text.split('\n').filter(line => line.includes('|'));
    if (lines.length < 2) return <p className="text-sm">{text}</p>;

    const rows = lines.map(line => 
      line.split('|').map(cell => cell.trim()).filter(cell => cell)
    );

    // Check if second row is separator (markdown style)
    const hasSeparator = rows[1]?.every(cell => cell.match(/^-+$/));
    const headerRows = hasSeparator ? [rows[0]] : [];
    const bodyRows = hasSeparator ? rows.slice(2) : rows;

    return (
      <div className="overflow-x-auto my-3 border rounded-lg">
        <table className="min-w-full divide-y divide-border text-sm">
          {headerRows.length > 0 && (
            <thead className="bg-muted">
              <tr>
                {headerRows[0].map((cell, idx) => (
                  <th key={idx} className="px-3 py-2 text-left font-semibold">
                    {cell}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          <tbody className="divide-y divide-border">
            {bodyRows.map((row, rowIdx) => (
              <tr key={rowIdx} className="hover:bg-muted/50">
                {row.map((cell, cellIdx) => (
                  <td key={cellIdx} className="px-3 py-2">
                    {formatInlineElements(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="relative group">
      <div className="prose prose-sm max-w-none">
        {formatText(content)}
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-0 right-0 h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={handleCopy}
      >
        {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
      </Button>
    </div>
  );
};

const ChatInterface: React.FC<ChatInterfaceProps> = ({ selectedDocuments }) => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(() => {
    const id = crypto.randomUUID();
    console.log('Generated session ID:', id);
    return id;
  });
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const clearChat = () => {
    setMessages([]);
    setSessionId(crypto.randomUUID()); // Generate new session ID when clearing chat
  };

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
    
    console.log('Sending message:', input);

    // Remove artificial delay - streaming should start quickly

    try {
      const response = await fetch('/api/chat/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input,
          similarity_top_k: 10,
          vector_distance_threshold: 0.5,
          document_ids: selectedDocuments.length > 0 ? selectedDocuments.map(doc => doc.id) : null,
          session_id: sessionId,
        }),
      });

      if (!response.ok) throw new Error('Failed to send message');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      // Don't add the assistant message yet - let the thinking animation show
      let assistantMessage: ChatMessage = {
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      };
      
      let messageAdded = false;

      if (reader) {
        let buffer = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;
          
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep the last incomplete line in the buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                console.log('Received SSE data:', data, 'Content:', data.content);
                if (data.content) {
                  assistantMessage.content += data.content;
                  console.log('Updated message content:', assistantMessage.content);
                  
                  if (!messageAdded) {
                    // Add the assistant message on first content
                    console.log('Adding first message');
                    flushSync(() => {
                      setMessages((prev) => [...prev, assistantMessage]);
                    });
                    messageAdded = true;
                  } else {
                    // Update existing message
                    console.log('Updating existing message');
                    flushSync(() => {
                      setMessages((prev) => {
                        const newMessages = [...prev];
                        const lastMessage = newMessages[newMessages.length - 1];
                        newMessages[newMessages.length - 1] = { 
                          ...lastMessage,
                          content: assistantMessage.content,
                          isStreaming: true 
                        };
                        console.log('New message state:', newMessages[newMessages.length - 1]);
                        return newMessages;
                      });
                    });
                  }
                  
                  // Remove artificial delay for smoother streaming
                } else if (data.done) {
                  console.log('Stream completed');
                  // Handle completion signal
                  break;
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e);
              }
            }
          }
        }
      }
      // Mark streaming as complete
      if (messageAdded) {
        setMessages((prev) => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = { 
            ...newMessages[newMessages.length - 1],
            isStreaming: false 
          };
          return newMessages;
        });
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

  return (
    <>
      <div className="fixed top-4 right-4 z-50">
        <Button
          onClick={() => setOpen(true)}
          className="relative rounded-full h-12 w-12 shadow-lg transition-all duration-200 hover:scale-105"
          size="icon"
        >
          <img src="/logos/pineapple.png" alt="Chat" className="h-6 w-6" />
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
        <SheetContent className="w-[400px] sm:w-[600px] lg:w-[700px] flex flex-col p-0">
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
                      {doc.display_name.length > 20 
                        ? `${doc.display_name.slice(0, 20)}...` 
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
                  <img src="/logos/pineapple.png" alt="Chat" className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>Ask questions about your documents</p>
                  <p className="text-xs mt-2">Tables, rates, and data will be formatted clearly</p>
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
                      className={`max-w-[85%] rounded-lg px-4 py-3 transition-all duration-200 ${
                        message.role === 'user'
                          ? 'bg-primary text-primary-foreground shadow-sm'
                          : 'bg-muted shadow-sm hover:shadow-md'
                      }`}
                    >
                      {message.role === 'assistant' ? (
                        message.content ? (
                          <>
                            {formatMessageContent(message.content)}
                            {message.isStreaming && (
                              <span className="inline-block w-1 h-4 bg-current animate-pulse ml-0.5" />
                            )}
                          </>
                        ) : (
                          <div className="space-y-2">
                            <Skeleton className="h-4 w-[200px]" />
                            <Skeleton className="h-4 w-[160px]" />
                          </div>
                        )
                      ) : (
                        <p className="text-sm">{message.content}</p>
                      )}
                    </div>
                  </div>
                ))
              )}
              {isLoading && (messages[messages.length - 1]?.role !== 'assistant' || !messages[messages.length - 1]?.content) && (
                <div className="flex justify-start animate-in fade-in-50 slide-in-from-bottom-2 duration-300">
                  <div className="bg-muted rounded-lg px-4 py-3 shadow-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-sm italic text-muted-foreground">
                        Thinking<ThinkingDots />
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
          
          <div className="border-t p-4 bg-background/95 backdrop-blur-sm">
            <div className="flex gap-2">
              <Input
                id="chat-input"
                name="chat-input"
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