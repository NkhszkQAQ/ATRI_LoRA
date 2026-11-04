import { useEffect, useState } from 'react';
import { Button, Drawer, Input, Message, Select, Space, Table, Tag, Typography } from '@arco-design/web-react';
import dayjs from 'dayjs';
import { api } from '../../api';

const sevColor: Record<string, string> = {
  low: 'gray',
  medium: 'orangered',
  high: 'red',
  critical: 'magenta',
};

const catLabel: Record<string, string> = {
  self_harm: '自伤',
  emotional_crisis: '情绪危机',
  prompt_extract: '套取提示词',
  jailbreak: '越狱',
  other: '其他',
};

export default function AdminSafety() {
  const [rows, setRows] = useState<any[]>([]);
  const [category, setCategory] = useState<string>('');
  const [userQ, setUserQ] = useState('');
  const [detail, setDetail] = useState<any>(null);

  const load = async () => {
    const p = new URLSearchParams();
    if (category) p.set('category', category);
    if (userQ.trim()) p.set('user', userQ.trim());
    setRows(await api.adminSafety(p));
  };

  useEffect(() => {
    load().catch((e) => Message.error(e.message));
  }, []);

  return (
    <div>
      <h2>风控 / 安全事件</h2>
      <p style={{ color: 'var(--muted)' }}>
        展示用户名、IP、UA、模型思考与回复、Token 元信息；查看原文会记入审计。
      </p>
      <Space wrap style={{ marginBottom: 12 }}>
        <Input
          placeholder="用户名 / 邮箱"
          value={userQ}
          onChange={setUserQ}
          style={{ width: 200 }}
          allowClear
        />
        <Select
          placeholder="类别"
          allowClear
          style={{ width: 180 }}
          value={category || undefined}
          onChange={(v) => setCategory(v || '')}
          options={[
            { label: '自伤', value: 'self_harm' },
            { label: '情绪危机', value: 'emotional_crisis' },
            { label: '套取提示词', value: 'prompt_extract' },
            { label: '越狱', value: 'jailbreak' },
          ]}
        />
        <Button type="primary" onClick={load}>
          筛选
        </Button>
      </Space>
      <Table
        rowKey="id"
        data={rows}
        scroll={{ x: 1100 }}
        columns={[
          {
            title: '用户',
            dataIndex: 'user_name',
            width: 140,
            render: (v, r) => (
              <div>
                <div>{v || '未知'}</div>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {r.user_email}
                </Typography.Text>
              </div>
            ),
          },
          {
            title: '类别',
            dataIndex: 'category',
            width: 110,
            render: (v) => catLabel[v] || v,
          },
          {
            title: '严重度',
            dataIndex: 'severity',
            width: 90,
            render: (v) => <Tag color={sevColor[v] || 'gray'}>{v}</Tag>,
          },
          { title: '动作', dataIndex: 'action', width: 90 },
          { title: 'IP', dataIndex: 'ip', width: 120, ellipsis: true },
          {
            title: '请求摘要',
            dataIndex: 'raw_text',
            ellipsis: true,
            render: (v) => (v || '').slice(0, 48),
          },
          {
            title: '时间',
            dataIndex: 'created_at',
            width: 130,
            render: (v) => dayjs(v).format('MM-DD HH:mm'),
          },
          {
            title: '详情',
            width: 90,
            fixed: 'right',
            render: (_, r) => (
              <Button
                size="mini"
                type="primary"
                onClick={async () => {
                  Message.warning('查看详情已记入审计日志');
                  setDetail(await api.adminSafetyDetail(r.id));
                }}
              >
                查看
              </Button>
            ),
          },
        ]}
      />
      <Drawer
        title="风控事件详情"
        visible={!!detail}
        onCancel={() => setDetail(null)}
        footer={null}
        width={640}
      >
        {detail && (
          <div className="admin-detail">
            <section>
              <h4>用户</h4>
              <p>
                <b>{detail.user_name}</b>
                <br />
                <Typography.Text type="secondary">{detail.user_email}</Typography.Text>
              </p>
            </section>
            <section>
              <h4>请求环境</h4>
              <p>IP：{detail.ip || '—'}</p>
              <p style={{ wordBreak: 'break-all' }}>UA：{detail.user_agent || '—'}</p>
              <p>模型：{detail.model || '—'}</p>
              <p>对话：{detail.conversation_title || '—'} {detail.conversation_id ? `(#${detail.conversation_id})` : ''}</p>
              <p>
                类别 / 严重度 / 概率：{catLabel[detail.category] || detail.category} / {detail.severity} /{' '}
                {detail.probability}%
              </p>
              <p>动作：{detail.action} · 语言：{detail.language_hint}</p>
              <p>时间：{dayjs(detail.created_at).format('YYYY-MM-DD HH:mm:ss')}</p>
            </section>
            <section>
              <h4>用户请求原文</h4>
              <pre className="admin-pre">{detail.raw_text || '—'}</pre>
            </section>
            <section>
              <h4>模型思考</h4>
              <pre className="admin-pre think">{detail.reasoning_text || '（无）'}</pre>
            </section>
            <section>
              <h4>模型回复</h4>
              <pre className="admin-pre">{detail.reply_text || '—'}</pre>
            </section>
            <section>
              <h4>Token / 鉴权信息</h4>
              <pre className="admin-pre">
                {detail.token_info ? JSON.stringify(detail.token_info, null, 2) : '（无）'}
              </pre>
            </section>
          </div>
        )}
      </Drawer>
      <style>{`
        .admin-detail section { margin-bottom: 18px; }
        .admin-detail h4 { margin: 0 0 8px; font-size: 14px; }
        .admin-pre {
          white-space: pre-wrap;
          word-break: break-word;
          background: #f7f4ee;
          border: 1px solid #e5ddd0;
          border-radius: 8px;
          padding: 10px 12px;
          font-size: 13px;
          max-height: 240px;
          overflow: auto;
          margin: 0;
        }
        .admin-pre.think { color: #5c564c; background: #f1ebe0; }
      `}</style>
    </div>
  );
}
