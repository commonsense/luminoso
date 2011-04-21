
$ ->
  # Canvas init
  window.viewer = new Viewer(document.getElementById("canvas"))
  window.viewer.setData('pldb')
  null

cmap = ([i, 0, 255-i] for i in [0...256])
MIX = .5
light_cmap = ([255*MIX + (1-MIX)*i, 0, 255*MIX + (1-MIX)*(255-i)] for i in [0...256])

class window.Viewer
  constructor: (canvas) ->
    @projection = new Projection()
    @running = false
    @ctx = null
    @imgData = null
    @canvas = canvas
    @lastInteraction = null
    @timeout = 3000
    @points = null
    @colors = null

  init: (data) ->
    @points = data.points
    @names = data.names
    @colors = (Math.floor(255*Math.random()) for i in [0...@points.length])
    @projection.reset()
    w = @canvas.width; h = @canvas.height
    @ctx = @canvas.getContext "2d"
    @ctx.clearRect(0, 0, w, h);
    @imgData = @ctx.getImageData(0, 0, w, h)
    @running = true
    $(@canvas).mousemove $.proxy(@onMouse, this)
    @lastInteraction = new Date
    @refresh()
    null

  refresh: ->
    requestAnimFrame($.proxy(@tick, this), @canvas)

  _getBinaryAjax: (url, callback) ->
    xhr = new XMLHttpRequest()
    xhr.open('GET', url, true)
    xhr.overrideMimeType 'text/plain; charset=x-user-defined'
    xhr.onreadystatechange = ->
      if xhr.readyState == 4
        callback(parseCoords(xhr.responseText))
    xhr.send null

  setData: (url) ->
    data = {}
    maybeInit = =>
      @init data if (data.points? and data.names?)
    
    @_getBinaryAjax url+'.coords', (coords) =>
      data.points = coords.data
      maybeInit()

    $.ajax
      url: url+'.names'
      dataType: 'text'
      success: (names) =>
        data.names = names.split('\n')
        maybeInit()

  onMouse: (e) ->
    @lastInteraction = new Date
    cosine = Math.cos((e.clientX - 450)/300)
    @projection.x_vec = [1, cosine, 1-cosine]
    @projection.y_vec = [0, 1-cosine, -cosine]
    @projection.zoom = 3*e.clientY
    if (!@running)
      console.log('resuming')
      @refresh()
    null

  tick: ->
    # Cache really common references
    project = @projection.makeProjector()
    arr = @arr; points = @points; colors = @colors
    w = @imgData.width; h = @imgData.height; arr = @imgData.data
    for offset in [0...w*h*4]
      arr[offset] = 255

    start = new Date
    for i in [0...points.length]
      vec = points[i]
      cidx = colors[i]
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
    @ctx.putImageData(@imgData, 0, 0)
    drawing_took = new Date - start
    console.log("Drawing took", drawing_took) if (drawing_took > 33)

    start = new Date
    names = @names
    for i in [0...points.length/10]
      vec = points[i]
      cidx = colors[i]
      [x, y] = project vec
      @ctx.fillText(names[i], Math.floor(x), Math.floor(y))
    labeling_took = new Date - start
    console.log("Labeling ", points.length/10, "points took", labeling_took) if (labeling_took > 15)

    if new Date - @lastInteraction > @timeout
      console.log('stopping')
      @running = false
    else
      @running = true
      @refresh()
    null


class Projection
  constructor: ->
    @reset

  reset: ->
    @x_vec = [1, 0, 0]
    @y_vec = [0, 1, 0]
    @origin_x = 0
    @origin_y = 200
    @zoom = 100

  makeProjector: ->
    x_vec = @x_vec; y_vec = @y_vec
    origin_x = @origin_x; origin_y = @origin_y
    zoom = @zoom
    return (vec) ->
      x = dot(x_vec, vec); y = dot(y_vec, vec)
      [x*zoom+origin_x, y*zoom+origin_y]

  project: (vec) ->
    (@makeProjector()) vec


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

parseCoords = (data) ->
  ndim = readUInt32(data, 0)
  nitems = readUInt32(data, 4)
  offset = 8
  arr = []
  for item in [0...nitems] by 1
    entry = new Float32Array(ndim)
    arr.push(entry)
    for dim in [0...ndim] by 1
      entry[dim] = readFloat32(data, offset)
      offset += 4
  return {ndim: ndim, nitems: nitems, data: arr}

dot = (vec1, vec2) ->
  s = 0
  for i in [0...vec1.length] by 1
    s += vec1[i] * vec2[i]
  s
