export default function NotificationsPage() {
  const notifications = [
    { icon: 'check_circle', title: 'Transfer Completed', desc: '$500.00 sent to Sarah Jenkins', time: '2 min ago', color: 'text-secondary', read: false },
    { icon: 'smart_toy', title: 'AI Agent Alert', desc: 'Unusual login attempt blocked from IP 192.168.1.45', time: '1 hour ago', color: 'text-error', read: false },
    { icon: 'verified_user', title: 'KYC Update', desc: 'Your identity verification has been approved', time: '3 hours ago', color: 'text-secondary', read: true },
    { icon: 'payments', title: 'Loan Payment Due', desc: 'Monthly payment of $425.00 due in 3 days', time: 'Yesterday', color: 'text-on-surface-variant', read: true },
    { icon: 'security', title: 'Privacy Policy Updated', desc: 'We\'ve updated our data retention policy. Review changes.', time: '2 days ago', color: 'text-tertiary', read: true },
    { icon: 'account_balance_wallet', title: 'Salary Deposited', desc: '$5,400.00 received from TechCorp Inc.', time: '3 days ago', color: 'text-secondary', read: true },
  ];

  return (
    <div className="p-[20px] md:p-[40px] max-w-[800px] mx-auto">
      <div className="flex items-center justify-between mb-[24px]">
        <div>
          <h1 className="font-[var(--font-headline)] text-[24px] md:text-[32px] font-semibold text-on-surface tracking-tight">Notifications</h1>
          <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px]">Stay updated with your account activity</p>
        </div>
        <button className="font-[var(--font-mono)] text-[12px] text-secondary bg-transparent border-none cursor-pointer tracking-[0.05em]">Mark all read</button>
      </div>

      <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 overflow-hidden">
        {notifications.map((n, i) => (
          <div key={i} className={`flex items-start gap-[14px] py-[16px] px-[20px] border-b border-outline-variant/10 last:border-none hover:bg-surface-container-low/30 transition-colors cursor-pointer ${!n.read ? 'bg-secondary-container/5' : ''}`}>
            <div className={`w-[36px] h-[36px] rounded-full flex items-center justify-center shrink-0 ${!n.read ? 'bg-secondary-container/30' : 'bg-surface-container-high'}`}>
              <span className={`material-symbols-outlined ${n.color}`} style={{ fontSize: '18px', fontVariationSettings: !n.read ? "'FILL' 1" : '' }}>{n.icon}</span>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-[8px]">
                <p className="font-[var(--font-body)] text-[14px] text-on-surface font-medium">{n.title}</p>
                {!n.read && <div className="w-[6px] h-[6px] rounded-full bg-secondary shrink-0"></div>}
              </div>
              <p className="font-[var(--font-body)] text-[13px] text-on-surface-variant mt-[2px] truncate">{n.desc}</p>
            </div>
            <span className="font-[var(--font-mono)] text-[10px] text-outline tracking-[0.05em] whitespace-nowrap shrink-0">{n.time}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
