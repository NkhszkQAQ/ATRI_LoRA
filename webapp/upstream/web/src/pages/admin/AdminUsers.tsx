import { useEffect, useState } from 'react';
import { Button, Input, Message, Space, Table, Tag } from '@arco-design/web-react';
import { api } from '../../api';

export default function AdminUsers() {
  const [rows, setRows] = useState<any[]>([]);
  const [q, setQ] = useState('');

  const load = async () => setRows(await api.adminUsers(q || undefined));
  useEffect(() => {
    load().catch((e) => Message.error(e.message));
  }, []);

  return (
    <div>
      <h2>用户管理</h2>
      <Space style={{ marginBottom: 12 }}>
        <Input.Search placeholder="邮箱 / 显示名" value={q} onChange={setQ} onSearch={load} style={{ width: 260 }} />
        <Button onClick={load}>刷新</Button>
      </Space>
      <Table
        rowKey="id"
        data={rows}
        columns={[
          {
            title: '用户名',
            dataIndex: 'display_name',
            render: (v, r) => v || r.email?.split('@')[0] || '—',
          },
          { title: '邮箱', dataIndex: 'email' },
          {
            title: '角色',
            dataIndex: 'role',
            width: 100,
            render: (v) => <Tag color={v === 'admin' ? 'orangered' : 'green'}>{v}</Tag>,
          },
          {
            title: '操作',
            width: 200,
            render: (_, r) => (
              <Space>
                <Button
                  size="mini"
                  onClick={async () => {
                    await api.adminPatchUser(r.id, { role: r.role === 'admin' ? 'user' : 'admin' });
                    Message.success('已更新角色');
                    load();
                  }}
                >
                  切换角色
                </Button>
                <Button
                  size="mini"
                  status="danger"
                  onClick={async () => {
                    await api.adminPatchUser(r.id, { is_active: false });
                    Message.success('已停用');
                    load();
                  }}
                >
                  停用
                </Button>
              </Space>
            ),
          },
        ]}
      />
    </div>
  );
}
