import { Navigate, Route, Routes } from 'react-router-dom';
import { getToken } from './api';
import AuthPage from './pages/AuthPage';
import ChatPage from './pages/ChatPage';
import AdminLayout from './pages/admin/AdminLayout';
import AdminUsers from './pages/admin/AdminUsers';
import AdminChats from './pages/admin/AdminChats';
import AdminSafety from './pages/admin/AdminSafety';
import AdminModels from './pages/admin/AdminModels';
import AdminSmtp from './pages/admin/AdminSmtp';
import AdminAudit from './pages/admin/AdminAudit';

function Private({ children }: { children: JSX.Element }) {
  if (!getToken()) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<AuthPage mode="login" />} />
      <Route path="/register" element={<AuthPage mode="register" />} />
      <Route path="/" element={<Private><ChatPage /></Private>} />
      <Route path="/admin" element={<Private><AdminLayout /></Private>}>
        <Route index element={<Navigate to="users" replace />} />
        <Route path="users" element={<AdminUsers />} />
        <Route path="chats" element={<AdminChats />} />
        <Route path="safety" element={<AdminSafety />} />
        <Route path="models" element={<AdminModels />} />
        <Route path="smtp" element={<AdminSmtp />} />
        <Route path="audit" element={<AdminAudit />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
