import { useEffect, useState } from 'react';
import { Button, DatePicker, Drawer, Input, Message, Space, Table, Typography } from '@arco-design/web-react';
import dayjs from 'dayjs';
import { api } from '../../api';

export default function AdminChats() {
  const [rows, setRows] = useState<any[]>([]);
  const [userQ, setUserQ] = useState('');
  const [model, setModel] = useState('');
  const [range, setRange] = useState<any[]>([]);
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState<any[]>([]);
  const [title, setTitle] = useState('');

  const load = async () => {
    const p = new URLSearchParams();
    if (userQ.trim()) p.set('user', userQ.trim());
    if (model) p.set('model', model);
    if (range?.[0]) p.set('date_from', range[0].toISOString());
    if (range?.[1]) p.set('date_to', range[1].toISOString());
    setRows(await api.adminConversations(p));
  };

  useEffect(() => {
    load().catch((e) => Message.error(e.message));
  }, []);

  return (
    <div>
      <h2>聊天记录</h2>
      <Space wrap style={{ marginBottom: 12 }}>
        <Input
          placeholder="用户名 / 邮箱"
          value={userQ}
          onChange={setUserQ}
          style={{ width: 180 }}
          allowClear
        />
        <Input placeholder="模型名" value={model} onChange={setModel} style={{ width: 160 }} allowClear />
        <DatePicker.RangePicker showTime onChange={(_, d) => setRange(d as any)} />
        <Button type="primary" onClick={load}>
          筛选
        </Button>
      </Space>
      <Table
        rowKey="id"
        data={rows}
        columns={[
          {
            title: '用户',
            dataIndex: 'user_name',
            width: 160,
            render: (v, r) => (
              <div>
                <div>{v || '未知'}</div>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {r.user_email}
                </Typography.Text>
              </div>
            ),
          },
          { title: '标题', dataIndex: 'title' },
          { title: '模型', dataIndex: 'model', width: 160 },
          {
            title: '更新',
            dataIndex: 'updated_at',
            width: 160,
            render: (v) => dayjs(v).format('YYYY-MM-DD HH:mm'),
          },
          {
            title: '操作',
            width: 140,
            render: (_, r) => (
              <Button
                size="mini"
                onClick={async () => {
                  Message.info('查看原文将写入审计日志');
                  setTitle(`${r.user_name || '用户'} · ${r.title || '对话'}`);
                  setMsgs(await api.adminMessages(r.id));
                  setOpen(true);
                }}
              >
                查看（记审计）
              </Button>
            ),
          },
        ]}
      />
      <Drawer width={560} title={title || '对话原文'} visible={open} onCancel={() => setOpen(false)} footer={null}>
        {msgs.map((m) => (
          <div key={m.id} style={{ marginBottom: 14 }}>
            <b>{m.role === 'user' ? '用户' : m.role === 'assistant' ? 'Atri' : m.role}</b>
            {m.reasoning && (
              <pre
                style={{
                  whiteSpace: 'pre-wrap',
                  background: '#f1ebe0',
                  color: '#5c564c',
                  padding: 8,
                  borderRadius: 8,
                  fontSize: 12,
                }}
              >
                【思考】{m.reasoning}
              </pre>
            )}
            <pre style={{ whiteSpace: 'pre-wrap', background: '#fff', padding: 8, borderRadius: 8 }}>{m.content}</pre>
          </div>
        ))}
      </Drawer>
    </div>
  );
}
