import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  Handle,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
// Keep React referenced so classic JSX runtime works even without the Vite React plugin
void React;

// --- Mappers ---
function mapFields(properties) {
  if (!properties) return [];
  return properties.map((name) => ({
    displayName: name,
    name,
    type: "String", // default since no explicit types are provided
    subType: undefined,
    isGenerated: false,
    isPrimaryKey: name.toLowerCase().includes("id"),
    isRelation: false,
    isUnique: false,
    isRequired: false,
    isReadOnly: false,
  }));
}

function mapEdgesFromRelations(relationTypes, entities) {
  if (!Array.isArray(relationTypes) || !Array.isArray(entities)) return []
  const lower = (s) => String(s || '').toLowerCase()
  const edges = []
  const edgeIds = new Set()

  relationTypes.forEach((r, idx) => {
    const rName = r?.name || `rel_${idx}`
    const nameL = lower(rName)
    const matches = entities.filter((e) => nameL.includes(lower(e.name)))

    let source = null
    let target = null
    if (matches.length >= 2) {
      ;[source, target] = [matches[0], matches[1]]
    } else if (matches.length === 1) {
      target = matches[0]
      source = entities.find((e) => e.name !== target.name) || target
    } else {
      // no clear mapping; skip to avoid noise
      return
    }

    if (!source || !target || source.name === target.name) return
    const id = `rel__${rName}__${source.name}__${target.name}`
    if (edgeIds.has(id)) return
    edgeIds.add(id)
    edges.push({
      id,
      source: source.name,
      target: target.name,
      label: rName,
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed },
    })
  })
  return edges
}

function mapToRF(schema) {
  const entity_types = schema?.entity_types || []
  const relation_types = schema?.relation_types || []

  const count = entity_types.length
  const cols = Math.max(1, Math.ceil(Math.sqrt(count)))
  const gapX = 260
  const gapY = 160

  const nodes = entity_types.map((e, i) => {
    const col = i % cols
    const row = Math.floor(i / cols)
    return {
      id: e.name,
      position: { x: col * gapX, y: row * gapY },
      type: 'entity',
      data: { label: e.name, description: e.description || '', fields: mapFields(e.property_names) },
      style: { width: 240 },
    }
  })

  let edges = mapEdgesFromRelations(relation_types, entity_types)
  // Fallback 1: if relations exist but couldn't be mapped, distribute them across consecutive nodes
  if (edges.length === 0 && relation_types.length > 0 && entity_types.length > 1) {
    const n = entity_types.length
    edges = relation_types.map((r, i) => {
      const a = entity_types[i % n].name
      const b = entity_types[(i + 1) % n].name
      const label = r?.name || `rel_${i}`
      return {
        id: `auto_rel__${label}__${a}__${b}`,
        source: a,
        target: b,
        label,
        type: 'smoothstep',
        animated: true,
        markerEnd: { type: MarkerType.ArrowClosed },
      }
    })
  }
  // Fallback 2: no relations at all, but multiple nodes → at least connect sequentially without labels
  if (edges.length === 0 && relation_types.length === 0 && entity_types.length > 1) {
    edges = entity_types.slice(0, -1).map((e, i) => ({
      id: `auto__${e.name}__${entity_types[i + 1].name}`,
      source: e.name,
      target: entity_types[i + 1].name,
      type: 'smoothstep',
      animated: true,
      markerEnd: { type: MarkerType.ArrowClosed },
    }))
  }

  return { nodes, edges }
}

function EntityNode({ data }) {
  const { label, description, fields = [] } = data || {}
  return (
    <div className="rf-entity">
      <Handle type="target" position={Position.Left} id="l" />
      <div className="rf-entity-title">{label}</div>
      {description ? <div className="rf-entity-desc">{description}</div> : null}
      {Array.isArray(fields) && fields.length > 0 ? (
        <div className="rf-entity-fields">
          {fields.map((f, idx) => (
            <div key={idx} className="rf-field">
              <span className="rf-field-name">{typeof f === 'string' ? f : f?.displayName || f?.name}</span>
              {typeof f !== 'string' && f?.type ? (
                <span className="rf-field-type">{f.type}{f.subType ? `<${f.subType}>` : ''}</span>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      <Handle type="source" position={Position.Right} id="r" />
    </div>
  )
}

const API_BASE = import.meta.env.VITE_API_BASE || ''


function SchemaBlock({ schema }) {
  if (!schema) return null
  const initial = mapToRF(schema)
  const [nodes, setNodes] = useState(initial.nodes)
  const [edges, setEdges] = useState(initial.edges)
  const [selected, setSelected] = useState({ nodes: [], edges: [] })
  const [exportText, setExportText] = useState('')

  useEffect(() => {
    const next = mapToRF(schema)
    setNodes(next.nodes)
    setEdges(next.edges)
    setExportText('')
  }, [schema])

  const onNodesChange = (changes) => setNodes((nds) => applyNodeChanges(changes, nds))
  const onEdgesChange = (changes) => setEdges((eds) => applyEdgeChanges(changes, eds))
  const onConnect = (connection) =>
    setEdges((eds) => addEdge({ ...connection, type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed } }, eds))

  const EntityNodeWrapper = useCallback(({ data, id }) => (
    <EntityNode
      data={{
        ...data,
        onEdit: () => {
          const name = prompt('Entity name:', data.label || id) || data.label || id
          const fieldsStr = prompt(
            'Fields (comma-separated):',
            (data.fields || [])
              .map((f) => (typeof f === 'string' ? f : f?.displayName || f?.name))
              .join(', ')
          ) || ''
          const fields = fieldsStr.split(',').map((s) => s.trim()).filter(Boolean)
          setNodes((nds) => nds.map((n) => (n.id === id ? { ...n, data: { ...n.data, label: name, fields } } : n)))
        },
      }}
    />
  ), [setNodes])

  const nodeTypes = useMemo(() => ({ entity: EntityNodeWrapper }), [EntityNodeWrapper])

  const addEntity = () => {
    const base = 'Entity'
    let i = nodes.length + 1
    let id = `${base}${i}`
    const ids = new Set(nodes.map((n) => n.id))
    while (ids.has(id)) { i += 1; id = `${base}${i}` }
    setNodes((nds) => ([
      ...nds,
      { id, type: 'entity', data: { label: id, description: '', fields: [] }, position: { x: (nds.length % 4) * 260, y: Math.floor(nds.length / 4) * 160 }, style: { width: 240 } },
    ]))
  }

  const deleteSelected = () => {
    const selNodeIds = new Set(selected.nodes.map((n) => n.id))
    const selEdgeIds = new Set(selected.edges.map((e) => e.id))
    setEdges((eds) => eds.filter((e) => !selEdgeIds.has(e.id) && !selNodeIds.has(e.source) && !selNodeIds.has(e.target)))
    setNodes((nds) => nds.filter((n) => !selNodeIds.has(n.id)))
  }

  const exportSchema = () => {
    const entity_types = nodes.map((n) => ({
      name: n.data?.label || n.id,
      description: n.data?.description || '',
      property_names: (n.data?.fields || []).map((f) => (typeof f === 'string' ? f : f?.displayName || f?.name)).filter(Boolean),
    }))
    const relation_types = edges.map((e) => ({ name: e.label || `${e.source}_${e.target}`, property_names: [] }))
    const exported = { entity_types, relation_types }
    const pretty = JSON.stringify(exported, null, 2)
    setExportText(pretty)
    console.log('Exported schema:', exported)
  }

  return (
    <div className="schema-block" style={{ width: '100%', height: 560 }}>
      <div className="schema-toolbar">
        <button className="btn" onClick={addEntity}>Add Entity</button>
        <button className="btn" onClick={deleteSelected} disabled={!selected.nodes.length && !selected.edges.length}>Delete Selected</button>
        <button className="btn" onClick={exportSchema}>Export Schema</button>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={(sel) => setSelected({ nodes: sel?.nodes || [], edges: sel?.edges || [] })}
      >
        <MiniMap />
        <Controls />
        <Background gap={12} />
      </ReactFlow>
      {exportText ? <pre className="schema-json" style={{ marginTop: 8 }}>{exportText}</pre> : null}
    </div>
  )
}

function Message({ role, text }) {
  const isUser = role === 'user'
  // Extract a JSON object from the assistant message, if present
  const parseSchemaFromText = (t) => {
    if (!t || isUser) return { prefix: t, schema: null }
    const firstBrace = t.indexOf('{')
    const lastBrace = t.lastIndexOf('}')
    if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
      return { prefix: t, schema: null }
    }
    const candidate = t.slice(firstBrace, lastBrace + 1)
    try {
      const obj = JSON.parse(candidate)
      const prefix = t.slice(0, firstBrace).trimEnd()
      return { prefix, schema: obj }
    } catch {
      return { prefix: t, schema: null }
    }
  }
  const { prefix, schema } = parseSchemaFromText(text)
  return (
    <>
      <div className={`msg ${isUser ? 'user' : 'bot'}`}>
        <div className="bubble">
          <span className="role">{isUser ? 'You' : 'Assistant'}</span>
          <div className="text">{prefix || (text || '')}</div>
        </div>
      </div>
      {!isUser && schema && (
        <div className="schema-row">
          <SchemaBlock schema={schema} />
        </div>
      )}
    </>
  )
}

export default function App() {
  const [messages, setMessages] = useState([
    { id: 1, role: 'bot', text: 'Hi! How can I help you today?' },
  ])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState('')
  const nextId = useRef(2)
  const scrollerRef = useRef(null)
  const conversationId = useMemo(() =>
    (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now() + Math.random()),
    [])
  const fileInputRef = useRef(null)

  const pickFile = () => fileInputRef.current?.click()

  const onFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadMsg('')
    if (!API_BASE) {
      setUploadMsg('Set VITE_API_BASE to enable uploads.')
      e.target.value = ''
      return
    }
    try {
      setUploading(true)
      const url = `${API_BASE.replace(/\/$/, '')}/upload/`
      const form = new FormData()
      form.append('file', file, file.name)
      // include conversation_id so backend can associate the file
      form.append('conversation_id', conversationId)
      const res = await fetch(url, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setUploadMsg(`Analyzing ${data.filename} (${data.content_type})`)

      // Inform chat UI
      const msg = { id: nextId.current++, role: 'bot', text: `File uploaded: ${data.filename}` }
      setMessages((m) => [...m, msg])
      // Immediately ask backend to proceed with analysis
      void send('Please proceed with analyzing the uploaded document.')
    } catch (err) {
      setUploadMsg(`Upload failed: ${err.message}`)
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  useEffect(() => {
    const el = scrollerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const send = async (value) => {
    const trimmed = value.trim()
    if (!trimmed || busy) return
    const userMsg = { id: nextId.current++, role: 'user', text: trimmed }
    setMessages((m) => [...m, userMsg])
    setInput('')
    setBusy(true)

    // placeholder assistant message to stream into
    const assistantId = nextId.current++
    setMessages((m) => [...m, { id: assistantId, role: 'bot', text: '' }])

    // If API_BASE is set, stream from /chat; else use demo
    if (API_BASE) {
      try {
        const url = `${API_BASE.replace(/\/$/, '')}/chat`
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: trimmed, conversation_id: conversationId }),
        })
        if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let done = false
        while (!done) {
          const { value, done: rdDone } = await reader.read()
          done = rdDone
          if (value) {
            const chunk = decoder.decode(value)
            setMessages((m) => m.map(msg => msg.id === assistantId ? { ...msg, text: msg.text + chunk } : msg))
          }
        }
      } catch (e) {
        setMessages((m) => m.map(msg => msg.id === assistantId ? { ...msg, text: `API error: ${e.message}` } : msg))
      } finally {
        setBusy(false)
      }
      return
    }

    // Demo streaming fallback without backend
    const demo = `You said: "${trimmed}". This is a demo response.`
    let i = 0
    const interval = setInterval(() => {
      if (i >= demo.length) {
        clearInterval(interval)
        setBusy(false)
        return
      }
      const ch = demo[i++]
      setMessages((m) => m.map(msg => msg.id === assistantId ? { ...msg, text: msg.text + ch } : msg))
    }, 12)
  }

  const onSubmit = (e) => {
    e.preventDefault()
    send(input)
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send(input)
    }
  }

  return (
    <div className="chat-container">
      <header className="chat-header">KG Builder Chat</header>
      <div className="messages" ref={scrollerRef}>
        {messages.map((m) => (
          <Message key={m.id} role={m.role} text={m.text} />
        ))}
      </div>
      <div className="uploader">
        <input
          ref={fileInputRef}
          type="file"
          onChange={onFileChange}
          style={{ display: 'none' }}
        />
        <button className="upload-btn" onClick={pickFile} disabled={uploading}>
          {uploading ? 'Uploading…' : 'Upload File'}
        </button>
        <span className="upload-msg">{uploadMsg}</span>
      </div>
      <form className="composer" onSubmit={onSubmit}>
        <textarea
          className="input"
          placeholder={busy ? 'Thinking…' : 'Type your message'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={busy}
          rows={1}
        />
        <button className="send" type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
