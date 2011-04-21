
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
    w = @canvas.width; h = @canvas.height

    @projection = new Projection(data.ndim, w, h)
    @ctx = @canvas.getContext "2d"
    @imgData = @ctx.getImageData(0, 0, w, h)

    # Set up animation loop and timers
    @lastInteraction = new Date
    @refresh()
    $(@canvas).mousemove $.proxy(@onMouse, this)
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
      data.ndim = coords.ndim
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
    @projection.moveTowards @points[0], e.clientX, e.clientY
    if (!@running)
      console.log('resuming')
      @refresh()
    null

  tick: ->
    @projection.tick()

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
  constructor: (k, width, height) ->
    @k = k; @width = width; @height = height
    @x_vec = new Float32Array(@k)
    @y_vec = new Float32Array(@k)
    @x_tgt = new Float32Array(@k)
    @y_tgt = new Float32Array(@k)
    @reset()

  reset: ->
    for i in [0...@k]
      @x_vec[i] = @x_tgt[i] = 0
      @y_vec[i] = @y_tgt[i] = 0
    @x_tgt[0] = 1.0
    @y_tgt[1] = 1.0
    @origin_x = @width/2
    @origin_y = @height/2
    @zoom = 100

  tick: ->
    for i in [0...@k]
      @x_vec[i] = @x_vec[i] * .5 + @x_tgt[i] * .5
      @y_vec[i] = @y_vec[i] * .5 + @y_tgt[i] * .5

  # Move a k-space vector towards a screen-point target
  moveTowards: (vec, targetX, targetY) ->
    # Convert to projection coordinates
    targetX = (targetX - @origin_x)/@zoom
    targetY = (targetY - @origin_y)/@zoom

    # Find out where the point originally should have projected to.
    origX = dot(@x_tgt, vec); origY = dot(@y_tgt, vec)

    # delta = (target - orig) / dot(vec, vec)
    vecMag = dot(vec, vec)
    deltaX = (targetX - origX) / vecMag
    deltaY = (targetY - origY) / vecMag

    # target_matrix += delta[np.newaxis,:] * vec[:,np.newaxis]
    for i in [0...@k]
      @x_tgt[i] += deltaX * vec[i]
      @y_tgt[i] += deltaY * vec[i]

    magnitude = deltaX*deltaX + deltaY*deltaY
    power = Math.atan(magnitude)/10
    @orthogonalize power

  # Use normalization and Gram-Schmidt orthogonalization to ensure that
  # this projection represents a valid rotation of the component space,
  # not a dilation or skew.
  #
  # Actually, now it only does that when power=1. When power is less, the
  # space will not enforce strict rotations, it will only make the space
  # wobble back toward orthogonality.
  orthogonalize: (power) ->
    targetX = @x_tgt; targetY = @y_tgt
    prevX = copyVec(targetX); prevY = copyVec(targetY)
    normalizeInPlace(targetX)
    commonMag = dot(targetX, targetY) # NOTE: in the Python code the magnitude was y_vec, not y_tgt
    for i in [0...@k]
      targetY[i] -= targetX[i] * commonMag
    normalizeInPlace(targetY)
    # target = target*(power) + prev*(1-power)
    for i in [0...@k]
      targetX[i] = targetX[i]*power + prevX[i]*(1-power)
      targetY[i] = targetY[i]*power + prevY[i]*(1-power)

  makeProjector: ->
    x_vec = @x_vec; y_vec = @y_vec
    origin_x = @origin_x; origin_y = @origin_y
    zoom = @zoom
    return (vec) ->
      x = dot(x_vec, vec); y = dot(y_vec, vec)
      [x*zoom+origin_x, y*zoom+origin_y]

  project: (vec) ->
    (@makeProjector()) vec


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

normalizeInPlace = (vec) ->
  mag = Math.sqrt dot(vec, vec)
  for i in [0...vec.length]
    vec[i] /= mag
  null

copyVec = (vec) ->
  v2 = new Float32Array(vec.length)
  for i in [0...vec.length]
    v2[i] = vec[i]
  v2
