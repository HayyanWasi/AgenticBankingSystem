import { useState, useEffect } from 'react';
import * as api from '../utils/api';

interface PendingReview {
  user_id: number;
  full_name: string;
  id_card_num: string;
  loan_amount: number;
  loan_term_months: number;
  loan_purpose: string;
  type: "KYC" | "Loan";
  item_id: string;
}

export default function AdminPortal() {
  const [applications, setApplications] = useState<PendingReview[]>([]);
  const [loading, setLoading] = useState(true);
  const [processingId, setProcessingId] = useState<number | null>(null);
  const [notification, setNotification] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  // Fetch pending items from backend on component mount
  useEffect(() => {
    api.get("/api/v1/admin/pending-kyc")
      .then((data) => {
        setApplications(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load admin queue:", err);
        if (err.message?.includes('403')) {
          setErrorMsg("Access Denied: You do not have admin privileges.");
        } else {
          setErrorMsg("Failed to load admin queue.");
        }
        setLoading(false);
      });
  }, []);

  const handleReview = async (app: PendingReview, action: "approve" | "reject") => {
  setProcessingId(app.user_id);
  setNotification(''); 
  
  try {
    const res = await api.post("/api/v1/admin/review", {
      user_id: app.user_id,
      action: action,
      type: app.type,
      item_id: app.item_id
    });

    // Support both direct return and nested axios .data return
    const responseData = res.data ? res.data : res;

    if (responseData.status === "success") {
      // 1. Remove the user from the table visually
      setApplications(prev => prev.filter(item => item.item_id !== app.item_id));
      
      // 2. Display the exact notification message sent from your FastAPI backend
      setNotification(`Action Complete: ${responseData.message}`);
      
      // 3. Hide the notification after 3 seconds
      setTimeout(() => setNotification(''), 3000);
    } else {
      setNotification("Warning: Backend did not return a success status.");
    }
  } catch (err) {
    console.error("Review decision update failed:", err);
    setNotification("Error: Failed to communicate with the server.");
  } finally {
    setProcessingId(null);
  }
};

  if (loading) return <div className="p-10 font-[var(--font-mono)]">Loading Compliance Queue...</div>;

  if (errorMsg) return (
    <div className="p-[40px] max-w-[800px] mx-auto text-center mt-[40px]">
      <div className="w-[80px] h-[80px] bg-error-container text-error rounded-full flex items-center justify-center mx-auto mb-[24px]">
        <span className="material-symbols-outlined" style={{ fontSize: '40px' }}>gpp_bad</span>
      </div>
      <h1 className="font-[var(--font-headline)] text-[32px] font-bold text-on-surface mb-[12px]">Access Denied</h1>
      <p className="font-[var(--font-body)] text-[16px] text-on-surface-variant">{errorMsg}</p>
    </div>
  );

  return (
    <div className="p-[20px] md:p-[40px] max-w-[1200px] mx-auto">
      <div className="mb-[32px]">
        <h1 className="font-[var(--font-headline)] text-[28px] font-semibold text-on-surface">Compliance Admin Operations</h1>
        <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px]">Human-in-the-Loop verification queue for AI Flagged & Pending records</p>
      </div>
    {notification && (
  <div className="mb-[24px] p-[16px] bg-secondary-container text-on-secondary-container rounded-[8px] font-[var(--font-mono)] text-[14px] border border-secondary/20 animate-fade-in-up">
    <span className="material-symbols-outlined align-middle mr-2" style={{ fontSize: '18px' }}>info</span>
    {notification}
  </div>
)}
      {applications.length === 0 ? (
        <div className="bg-surface-container-lowest p-10 text-center rounded-[16px] border border-outline-variant/10">
          <span className="material-symbols-outlined text-outline" style={{ fontSize: '48px' }}>done_all</span>
          <p className="font-[var(--font-body)] text-[16px] text-on-surface-variant mt-2">The compliance queue is completely empty. Excellent work.</p>
        </div>
      ) : (
        <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 overflow-hidden">
          <table className="w-full border-collapse text-left">
  <thead>
    <tr className="bg-surface-container-high font-[var(--font-mono)] text-[12px] text-on-surface-variant border-b border-outline-variant/20">
      <th className="p-4">User Details</th>
      <th className="p-4">ID Card Number</th>
      <th className="p-4">Loan Details</th>
      <th className="p-4 text-right">Actions</th>
    </tr>
  </thead>
  <tbody>
    {applications.map((app) => (
      <tr key={app.item_id} className="border-b border-outline-variant/10 hover:bg-surface-bright transition-colors font-[var(--font-body)] text-[14px]">
        <td className="p-4">
          <div className="font-semibold text-on-surface">{app.full_name}</div>
          <div className="text-[12px] text-on-surface-variant">
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold mr-2 ${app.type === 'KYC' ? 'bg-primary-container text-on-primary-container' : 'bg-tertiary-container text-on-tertiary-container'}`}>
              {app.type}
            </span>
            User ID: {app.user_id}
          </div>
        </td>
        <td className="p-4 font-[var(--font-mono)] text-[13px]">{app.id_card_num}</td>
        
        <td className="p-4">
          {app.type === 'Loan' ? (
            <>
              <div className="font-[var(--font-mono)] font-semibold text-secondary">
                ${app.loan_amount} <span className="text-on-surface-variant text-[12px] font-normal">for {app.loan_term_months} months</span>
              </div>
              <div className="text-[12px] text-on-surface-variant mt-1 max-w-[200px] truncate" title={app.loan_purpose}>
                Purpose: {app.loan_purpose}
              </div>
            </>
          ) : (
            <div className="text-[12px] text-on-surface-variant mt-1">
               Identity Verification Only
            </div>
          )}
        </td>
        
        <td className="p-4 text-right">
          <div className="flex justify-end gap-2">
            <button 
              onClick={() => handleReview(app, "reject")}
              disabled={processingId === app.user_id}
              className="px-3 py-2 bg-surface-container-high text-error font-[var(--font-mono)] text-[12px] border-none rounded-[6px] cursor-pointer hover:bg-error/10 disabled:opacity-50"
            >
              Reject
            </button>
            <button 
              onClick={() => handleReview(app, "approve")}
              disabled={processingId === app.user_id}
              className="px-3 py-2 bg-secondary text-on-secondary font-[var(--font-mono)] text-[12px] border-none rounded-[6px] cursor-pointer hover:bg-on-surface-variant disabled:opacity-50"
            >
              {processingId === app.user_id ? "Saving..." : `Approve ${app.type}`}
            </button>
          </div>
        </td>
      </tr>
    ))}
  </tbody>
</table>
        </div>
      )}
    </div>
  );
}