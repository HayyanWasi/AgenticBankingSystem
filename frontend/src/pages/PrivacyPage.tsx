import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'ai';
  text: string;
  timestamp: string;
}

export default function PrivacyPage() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'ai', text: 'Hello! I\'m your AI Privacy Agent. I can help you understand our privacy policies, manage your data preferences, and answer questions about how your information is protected. What would you like to know?', timestamp: 'Just now' },
  ]);
  const [input, setInput] = useState('');
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const quickQuestions = [
    'How is my data stored?',
    'Who has access to my info?',
    'How do I delete my account?',
    'What data do you collect?',
  ];

  const aiResponses: Record<string, string> = {
    'How is my data stored?': 'Your data is stored using AES-256 encryption at rest and TLS 1.3 for data in transit. All sensitive information is segmented across geographically distributed servers with zero-knowledge architecture. Our AI agents process data in isolated environments with no persistent storage of raw personal data.',
    'Who has access to my info?': 'Access to your data follows a strict principle of least privilege. Only verified AI agents with specific task authorization can access relevant data segments. Human employees require multi-factor authentication and audit trails are maintained for every access event. You can view the full access log in your Security Dashboard.',
    'How do I delete my account?': 'You can request account deletion through Settings → Account → Delete Account. Per our policy, all personal data will be purged within 30 days. Certain anonymized transaction records may be retained for regulatory compliance (up to 7 years) but will contain no personally identifiable information.',
    'What data do you collect?': 'We collect: (1) Identity data — name, email, government ID for KYC. (2) Financial data — transaction history, account balances. (3) Behavioral data — login patterns for security. (4) Device data — IP, browser type for fraud prevention. You can control data sharing preferences in Settings → Privacy.',
  };

  const handleSend = (text?: string) => {
    const msg = text || input;
    if (!msg.trim()) return;

    const userMsg: Message = { role: 'user', text: msg, timestamp: 'Just now' };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    setTimeout(() => {
      const response = aiResponses[msg] || 'Thank you for your question. Based on our privacy policy, your data is protected under strict security protocols. I\'ve flagged this for a detailed review. Is there anything else you\'d like to know about your data privacy?';
      const aiMsg: Message = { role: 'ai', text: response, timestamp: 'Just now' };
      setMessages(prev => [...prev, aiMsg]);
    }, 800);
  };

  return (
    <div className="p-[20px] md:p-[40px] max-w-[900px] mx-auto h-[calc(100vh-40px)] md:h-[calc(100vh-80px)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-[20px] shrink-0">
        <div>
          <h1 className="font-[var(--font-headline)] text-[24px] md:text-[32px] font-semibold text-on-surface tracking-tight">AI Privacy Agent</h1>
          <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[2px]">Ask questions about your data privacy &amp; security</p>
        </div>
        <div className="flex items-center gap-[8px] bg-surface-container-high rounded-full px-[12px] py-[6px]">
          <span className="material-symbols-outlined text-secondary animate-pulse-glow" style={{ fontSize: '14px', fontVariationSettings: "'FILL' 1" }}>security</span>
          <span className="font-[var(--font-mono)] text-[11px] text-secondary tracking-[0.05em]">Privacy Agent Online</span>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 flex flex-col overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-[20px] flex flex-col gap-[16px]">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-[12px] animate-fade-in-up ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-[32px] h-[32px] rounded-full flex items-center justify-center shrink-0 ${
                msg.role === 'ai' ? 'bg-primary-container text-secondary' : 'bg-secondary text-on-secondary'
              }`}>
                <span className="material-symbols-outlined" style={{ fontSize: '16px', fontVariationSettings: "'FILL' 1" }}>
                  {msg.role === 'ai' ? 'smart_toy' : 'person'}
                </span>
              </div>
              <div className={`max-w-[75%] rounded-[16px] px-[16px] py-[12px] ${
                msg.role === 'ai' ? 'bg-surface-container-low' : 'bg-secondary text-on-secondary'
              }`}>
                <p className="font-[var(--font-body)] text-[14px] leading-[22px]">{msg.text}</p>
                <span className={`font-[var(--font-mono)] text-[10px] mt-[6px] block tracking-[0.05em] ${
                  msg.role === 'ai' ? 'text-on-surface-variant' : 'text-on-secondary/70'
                }`}>{msg.timestamp}</span>
              </div>
            </div>
          ))}
          <div ref={messagesEnd} />
        </div>

        {/* Quick Questions */}
        <div className="flex gap-[8px] px-[20px] py-[12px] overflow-x-auto border-t border-outline-variant/10">
          {quickQuestions.map((q) => (
            <button
              key={q}
              onClick={() => handleSend(q)}
              className="px-[12px] py-[6px] bg-surface-container-high rounded-full font-[var(--font-mono)] text-[11px] text-on-surface-variant tracking-[0.05em] whitespace-nowrap hover:bg-secondary-container hover:text-secondary transition-colors cursor-pointer border-none shrink-0"
            >
              {q}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="p-[16px] border-t border-outline-variant/10">
          <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="flex gap-[8px]">
            <input
              className="flex-1 bg-surface-bright border border-outline-variant rounded-[12px] py-[12px] px-[16px] font-[var(--font-body)] text-[14px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant"
              placeholder="Ask about your privacy..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button type="submit" className="w-[44px] h-[44px] bg-secondary text-on-secondary rounded-[12px] flex items-center justify-center cursor-pointer border-none hover:bg-on-surface-variant transition-colors shrink-0">
              <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>send</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
