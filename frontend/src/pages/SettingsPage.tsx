import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function SettingsPage() {
  const navigate = useNavigate();
  const [twoFa, setTwoFa] = useState(true);
  const [biometric, setBiometric] = useState(true);
  const [aiMonitoring, setAiMonitoring] = useState(true);
  const [emailNotif, setEmailNotif] = useState(true);
  const [pushNotif, setPushNotif] = useState(false);

  const Toggle = ({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) => (
    <label className="relative inline-flex items-center cursor-pointer">
      <input checked={value} onChange={() => onChange(!value)} className="sr-only peer" type="checkbox" />
      <div className="w-[44px] h-[24px] bg-outline-variant rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-surface-container-lowest after:border-outline-variant after:border after:rounded-full after:h-[20px] after:w-[20px] after:transition-all peer-checked:bg-secondary"></div>
    </label>
  );

  return (
    <div className="p-[20px] md:p-[40px] max-w-[800px] mx-auto">
      <div className="mb-[32px]">
        <h1 className="font-[var(--font-headline)] text-[24px] md:text-[32px] font-semibold text-on-surface tracking-tight">Settings</h1>
        <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px]">Manage your account preferences</p>
      </div>

      {/* Profile Section */}
      <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[24px] mb-[16px]">
        <h3 className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em] uppercase mb-[16px]">Profile</h3>
        <div className="flex items-center gap-[16px] mb-[20px]">
          <div className="w-[56px] h-[56px] rounded-full bg-primary-container border-2 border-secondary/30 overflow-hidden">
            <img
              alt="Profile"
              className="w-full h-full object-cover"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuDa3vfk84khTwpW8K7Mg7J5y_tcKxIJmTEPtfvNiLV1yljirvJtLBSNaNhKDMt4dJrzSPIv6_sfpO-wcRVx0SIRUbuqHnMjBXPy-eb2B7hf0PHPzJYPMbODQut5IqH5mFK5ZRtj96zK3fiY0v4bqQx1304AxGm1C2rQxSSisrqkfOXXes-81_dgXm6qGlBzNNddCyn0a13_9xALRgvvT-F66hFF0MnnaDKlN9Ad0BVtOF51dTpWtie8B5x1B0QU0jQ34Egk9l4y0MH9"
            />
          </div>
          <div>
            <p className="font-[var(--font-body)] text-[16px] text-on-surface font-medium">Alex Rivers</p>
            <p className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">alex.rivers@example.com</p>
          </div>
        </div>
        <button className="bg-surface-container-low text-on-surface font-[var(--font-mono)] text-[12px] tracking-[0.05em] px-[16px] py-[8px] rounded-[8px] cursor-pointer border-none hover:bg-surface-container-high transition-colors">Edit Profile</button>
      </div>

      {/* Security */}
      <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[24px] mb-[16px]">
        <h3 className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em] uppercase mb-[16px]">Security</h3>
        {[
          { label: 'Two-Factor Authentication', desc: 'Add extra security to your account', value: twoFa, onChange: setTwoFa },
          { label: 'Biometric Login', desc: 'Use fingerprint or face recognition', value: biometric, onChange: setBiometric },
          { label: 'AI Monitoring', desc: 'Let AI agents monitor for suspicious activity', value: aiMonitoring, onChange: setAiMonitoring },
        ].map((setting, i) => (
          <div key={i} className="flex items-center justify-between py-[12px] border-b border-outline-variant/10 last:border-none">
            <div>
              <p className="font-[var(--font-body)] text-[14px] text-on-surface font-medium">{setting.label}</p>
              <p className="font-[var(--font-body)] text-[13px] text-on-surface-variant">{setting.desc}</p>
            </div>
            <Toggle value={setting.value} onChange={setting.onChange} />
          </div>
        ))}
      </div>

      {/* Notifications */}
      <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[24px] mb-[16px]">
        <h3 className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em] uppercase mb-[16px]">Notifications</h3>
        {[
          { label: 'Email Notifications', desc: 'Receive alerts via email', value: emailNotif, onChange: setEmailNotif },
          { label: 'Push Notifications', desc: 'Browser push notifications', value: pushNotif, onChange: setPushNotif },
        ].map((setting, i) => (
          <div key={i} className="flex items-center justify-between py-[12px] border-b border-outline-variant/10 last:border-none">
            <div>
              <p className="font-[var(--font-body)] text-[14px] text-on-surface font-medium">{setting.label}</p>
              <p className="font-[var(--font-body)] text-[13px] text-on-surface-variant">{setting.desc}</p>
            </div>
            <Toggle value={setting.value} onChange={setting.onChange} />
          </div>
        ))}
      </div>

      {/* Danger Zone */}
      <div className="bg-surface-container-lowest rounded-[16px] border border-error/20 p-[24px]">
        <h3 className="font-[var(--font-mono)] text-[12px] text-error font-bold tracking-[0.05em] uppercase mb-[12px]">Danger Zone</h3>
        <p className="font-[var(--font-body)] text-[13px] text-on-surface-variant mb-[16px]">Permanently delete your account and all associated data.</p>
        <button className="bg-error text-on-error font-[var(--font-mono)] text-[12px] tracking-[0.05em] px-[16px] py-[8px] rounded-[8px] cursor-pointer border-none">Delete Account</button>
      </div>
    </div>
  );
}
