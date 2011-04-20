ctx = null
imgData = null
canvas = null
lastInteraction = null
timeout = 3000
window.running = false

$ ->
  # Canvas init
  canvas = document.getElementById("canvas")
  w=canvas.width; h=canvas.height
  ctx = canvas.getContext "2d"
  ctx.clearRect(0, 0, w, h);
  imgData = ctx.getImageData(0, 0, w, h)

  window.running = true
  $('#canvas').mousemove onMouse
  lastInteraction = new Date
  #setInterval tick, 50
  requestAnimFrame(tick, canvas)
  null

window.loadBinary = (url, callback) ->
  xhr = new XMLHttpRequest()
  xhr.open('GET', url, true)
  if (xhr.hasOwnProperty("responseType"))
    xhr.responseType = "arraybuffer"
  else
    xhr.overrideMimeType 'text/plain; charset=x-user-defined'
  xhr.onreadystatechange = =>
    if xhr.readyState == 4
      console.log xhr.status
      console.log xhr
      if xhr.hasOwnProperty('responseType') && xhr.responseType == 'arraybuffer'
        data = xhr.response
      else if xhr.mozResponseArrayBuffer?
        data = xhr.mozResponseArrayBuffer
      else
        alert 'Getting binary data unsupported in this browser.'
      callback data
  xhr.send null


points = ([[Math.random(), Math.random(), Math.random()], Math.floor(255*Math.random())] for i in [0...10000])
cmap = ([i, 0, 255-i] for i in [0...256])
MIX = .5
light_cmap = ([255*MIX + (1-MIX)*i, 0, 255*MIX + (1-MIX)*(255-i)] for i in [0...256])

dot = (vec1, vec2) ->
  s = 0
  for i in [0...vec1.length] by 1
    s += vec1[i] * vec2[i]
  s

x_vec = [1, 0, 0]
y_vec = [0, 1, 0]
origin_x = 0
origin_y = 200
zoom = 100

onMouse = (e) ->
  lastInteraction = new Date
  #console.log e.clientX
  cosine = Math.cos((e.clientX - 450)/300)
  x_vec = [1, cosine, 1-cosine]
  y_vec = [0, 1-cosine, -cosine]
  #console.log dot(x_vec, y_vec)
  zoom = 3*e.clientY
  #setTimeout tick, 0
  if (!running)
    console.log('resuming')
    requestAnimFrame(tick, canvas)
  null

project = (vec) ->
  x = dot(vec, x_vec); y = dot(vec, y_vec)
  [x*zoom+origin_x, y*zoom+origin_y]

window.tick = ->
  w = imgData.width; h = imgData.height; arr = imgData.data
  for offset in [0...w*h*4]
    arr[offset] = 255

  start = new Date
  for pt in points
    [vec, cidx] = pt
    [x, y] = project vec
    unless (0 < x < w and 0 < y < h)
      continue
    color = cmap[cidx]
    light_color = light_cmap[cidx]
    offset = 4*Math.floor((x + w*(Math.floor(y))));
    arr[offset] = color[0]; arr[offset+1] = color[1]; arr[offset+2] = color[2]; arr[offset+3] = 255;
    arr[offset-4] = light_color[0]; arr[offset-4+1] = light_color[1]; arr[offset-4+2] = light_color[2]; arr[offset-4+3] = 64;
    arr[offset+4] = light_color[0]; arr[offset+4+1] = light_color[1]; arr[offset+4+2] = light_color[2]; arr[offset+4+3] = 64;
    arr[offset-4*w] = light_color[0]; arr[offset-4*w+1] = light_color[1]; arr[offset-4*w+2] = light_color[2]; arr[offset-4*w+3] = 64;
    arr[offset+4*w] = light_color[0]; arr[offset+4*w+1] = light_color[1]; arr[offset+4*w+2] = light_color[2]; arr[offset+4*w+3] = 64;
  ctx.putImageData(imgData, 0, 0)
  drawing_took = new Date - start
  console.log("Drawing took", drawing_took) if (drawing_took > 33)

  start = new Date
  for i in [0...points.length/10]
    [vec, cidx] = points[i]
    [x, y] = project vec
    ctx.fillText("pt", Math.floor(x), Math.floor(y))
  labeling_took = new Date - start
  console.log("Labeling took", labeling_took) if (labeling_took > 15)

  if new Date - lastInteraction > timeout
    console.log('stopping')
    window.running = false
  else
    window.running = true
    requestAnimFrame(tick, canvas)
  null
