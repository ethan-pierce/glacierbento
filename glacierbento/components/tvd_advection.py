"""Models advection using a total-variation-diminishing (TVD) scheme.

The TVDAdvector models advective transport of a scalar field using a 
total-variation-diminishing (TVD) scheme. Here, we default to a Van Leer flux 
limiter, but subclasses can override the flux limiter method as needed. 