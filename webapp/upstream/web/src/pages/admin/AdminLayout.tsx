import { Outlet, useNavigate, useLocation, Link } from 'react-router-dom';
import { Menu } from '@arco-design/web-react';
import {
  IconUser,
  IconMessage,
  IconExclamationCircle,
  IconStorage,
  IconSettings,
  IconHistory,
  IconHome,
} from '@arco-design/web-react/icon';

const items = [
  { key: 'users', label: '用户管理', icon: <IconUser /> },
  { key: 'chats', label: '聊天记录', icon: <IconMessage /> },
  { key: 'safety', label: '安全事件', icon: <IconExclamationCircle /> },
  { key: 'models', label: '底层模型', icon: <IconStorage /> },
  { key: 'smtp', label: 'SMTP 设置', icon: <IconSettings /> },
  { key: 'audit', label: '审计日志', icon: <IconHistory /> },
];
export default function AdminLayout() {
  const nav = useNavigate();
  const loc = useLocation();
  const selected = loc.pathname.split('/').pop() || 'users';

  return (
    <div className="admin-layout">
      <aside className="admin-side">
        <div style={{ fontFamily: 'ZCOOL XiaoWei, serif', fontSize: 22, padding: '8px 10px 18px' }}>
          Atri Admin
        </div>
        <Menu
          theme="dark"
          selectedKeys={[selected]}
          onClickMenuItem={(k) => nav(`/admin/${k}`)}
          style={{ width: '100%' }}
        >
          {items.map((it) => (
            <Menu.Item key={it.key}>
              {it.icon} {it.label}
            </Menu.Item>
          ))}
        </Menu>
        <div style={{ marginTop: 'auto', padding: 12 }}>
          <Link to="/" style={{ color: '#9fd0c0' }}><IconHome /> 返回对话</Link>
        </div>
      </aside>
      <main className="admin-main">
        <Outlet />
      </main>
    </div>
  );
}
