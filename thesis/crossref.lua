-- Minimal cross-references for the thesis (no pandoc-crossref needed).
-- Numbers chapters/sections, figures and tables per chapter, prefixes captions
-- ("Figure 3.1: ...", "Table 4.2: ...") and replaces [@fig:x], [@tbl:x], [@sec:x]
-- (written as "Figure @fig:x" in the text) by the number. Runs before citeproc.

local labels = {}
local chapter = 0

local function strip_table_id(caption_blocks)
  -- "Table: text {#tbl:id}" -> pandoc keeps "{#tbl:id}" as text at the end.
  local id = nil
  for _, blk in ipairs(caption_blocks) do
    if blk.content then
      local n = #blk.content
      if n > 0 and blk.content[n].t == "Str" then
        local s = blk.content[n].text
        local m = s:match("^{#(tbl:[%w_%-]+)}$")
        if m then
          id = m
          blk.content:remove(n)
          if #blk.content > 0 and blk.content[#blk.content].t == "Space" then
            blk.content:remove(#blk.content)
          end
        end
      end
    end
  end
  return id
end

local function prefix(caption_blocks, word, num)
  if #caption_blocks == 0 then
    caption_blocks:insert(pandoc.Plain({}))
  end
  local first = caption_blocks[1]
  local pre = { pandoc.Strong({ pandoc.Str(word), pandoc.Space(), pandoc.Str(num .. ":") }), pandoc.Space() }
  for i = #pre, 1, -1 do first.content:insert(1, pre[i]) end
end

local function figure_id(fig)
  if fig.identifier and fig.identifier ~= "" then return fig.identifier end
  local id = nil
  fig:walk({ Image = function(img) if img.identifier ~= "" then id = id or img.identifier end end })
  return id
end

function Pandoc(doc)
  local counters = { 0, 0, 0, 0, 0, 0 }
  local nfig, ntbl = 0, 0
  for _, blk in ipairs(doc.blocks) do
    if blk.t == "Header" and not blk.classes:includes("unnumbered") then
      local lvl = blk.level
      counters[lvl] = counters[lvl] + 1
      for k = lvl + 1, #counters do counters[k] = 0 end
      if lvl == 1 then chapter = counters[1]; nfig, ntbl = 0, 0 end
      local parts = {}
      for k = 1, lvl do parts[#parts + 1] = tostring(counters[k]) end
      if blk.identifier ~= "" then labels[blk.identifier] = table.concat(parts, ".") end
    elseif blk.t == "Figure" then
      nfig = nfig + 1
      local num = chapter .. "." .. nfig
      local id = figure_id(blk)
      if id then labels[id] = num end
      prefix(blk.caption.long, "Figure", num)
    elseif blk.t == "Table" then
      -- full text width; column widths from the longest cell text (square root, so long
      -- columns get more room without squeezing short ones)
      local n = #blk.colspecs
      local maxlen, maxword = {}, {}
      for i = 1, n do maxlen[i] = 4; maxword[i] = 4 end
      local function scan(rows)
        for _, row in ipairs(rows) do
          for i, cell in ipairs(row.cells) do
            if i <= n then
              local txt = pandoc.utils.stringify(cell.contents)
              local len = utf8.len(txt) or 4
              if len > maxlen[i] then maxlen[i] = len end
              for w in txt:gmatch("%S+") do
                local wl = utf8.len(w) or #w
                if wl > maxword[i] then maxword[i] = wl end
              end
            end
          end
        end
      end
      scan(blk.head.rows)
      for _, body in ipairs(blk.bodies) do scan(body.body) end
      local total, w = 0, {}
      for i = 1, n do
        w[i] = math.max((math.min(maxword[i], 14) + 3) * 1.6, 4 * math.sqrt(math.min(maxlen[i], 120)))
        total = total + w[i]
      end
      for i = 1, n do blk.colspecs[i] = { blk.colspecs[i][1], w[i] / total } end
      ntbl = ntbl + 1
      local num = chapter .. "." .. ntbl
      local id = strip_table_id(blk.caption.long)
      if id then labels[id] = num; blk.identifier = id end
      prefix(blk.caption.long, "Table", num)
    end
  end
  doc = doc:walk({
    Cite = function(c)
      local out = {}
      for i, citation in ipairs(c.citations) do
        local key = citation.id
        if key:match("^fig:") or key:match("^tbl:") or key:match("^sec:") then
          out[#out + 1] = pandoc.Str(labels[key] or ("??" .. key))
          if i < #c.citations then out[#out + 1] = pandoc.Str(", ") end
        else
          return nil
        end
      end
      return out
    end
  })
  return doc
end
