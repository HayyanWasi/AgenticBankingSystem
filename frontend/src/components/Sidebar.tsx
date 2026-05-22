import { useState, useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { getDashboard } from '../utils/api';
const navItems = [
  { icon: 'dashboard', label: 'Dashboard', path: '/dashboard' },
  { icon: 'payments', label: 'Loans', path: '/loans' },
  { icon: 'receipt_long', label: 'Transactions', path: '/transactions' },
  { icon: 'swap_horiz', label: 'Transfer', path: '/transfer' },
  { icon: 'verified_user', label: 'KYC', path: '/kyc' },
  { icon: 'security', label: 'AI Privacy', path: '/privacy' },
];

const bottomItems = [
  { icon: 'notifications', label: 'Notifications', path: '/notifications' },
  { icon: 'settings', label: 'Settings', path: '/settings' },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    async function checkAdmin() {
      try {
        const userId = Number(localStorage.getItem('user_id'));
        if (userId) {
          const data = await getDashboard(userId);
          setIsAdmin(data.is_admin);
        }
      } catch (err) {
        console.error("Failed to check admin status:", err);
      }
    }
    checkAdmin();
  }, []);

  const dynamicNavItems = [...navItems];
  if (isAdmin) {
    dynamicNavItems.push({ icon: 'admin_panel_settings', label: 'Admin Portal', path: '/admin' });
  }

  return (
    <>
      {/* Desktop Sidebar */}
      <aside className="hidden md:flex flex-col w-[260px] min-h-screen bg-surface-container-lowest border-r border-outline-variant/20 p-[16px] justify-between shrink-0">
        {/* Logo */}
        <div>
          <button onClick={() => navigate('/dashboard')} className="flex items-center gap-[10px] mb-[32px] px-[8px] cursor-pointer bg-transparent border-none">
            <div className="w-[40px] h-[40px] bg-primary-container text-secondary rounded-full flex items-center justify-center">
              <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1", fontSize: '22px' }}>account_balance</span>
            </div>
            <span className="font-[var(--font-headline)] text-[18px] font-bold text-on-surface tracking-tight">Agentic Bank</span>
          </button>

          {/* Nav Items */}
          <nav className="flex flex-col gap-[4px]">
            {dynamicNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-[12px] px-[12px] py-[10px] rounded-[12px] font-[var(--font-mono)] text-[13px] font-medium tracking-[0.02em] transition-all duration-200 no-underline ${
                    isActive
                      ? 'bg-primary-container text-secondary'
                      : 'text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface'
                  }`
                }
              >
                <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>{item.icon}</span>
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Bottom Actions */}
        <div className="flex flex-col gap-[4px]">
          {bottomItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-[12px] px-[12px] py-[10px] rounded-[12px] font-[var(--font-mono)] text-[13px] font-medium tracking-[0.02em] transition-all duration-200 no-underline ${
                  isActive
                    ? 'bg-primary-container text-secondary'
                    : 'text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface'
                }`
              }
            >
              <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}

          <button
            onClick={() => navigate('/login')}
            className="flex items-center gap-[12px] px-[12px] py-[10px] rounded-[12px] font-[var(--font-mono)] text-[13px] font-medium tracking-[0.02em] transition-all duration-200 text-error hover:bg-error-container/30 cursor-pointer bg-transparent border-none w-full text-left"
          >
            <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>logout</span>
            Log Out
          </button>
        </div>
      </aside>

      {/* Mobile Bottom Bar */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-surface-container-lowest/90 backdrop-blur-[16px] border-t border-outline-variant/20 flex justify-around py-[8px] px-[4px]">
        {dynamicNavItems.slice(0, 5).map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex flex-col items-center gap-[2px] px-[8px] py-[4px] rounded-[8px] text-[10px] font-medium no-underline transition-colors ${
                isActive ? 'text-secondary' : 'text-on-surface-variant'
              }`
            }
          >
            <span className="material-symbols-outlined" style={{ fontSize: '22px' }}>{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>
    </>
  );
}
