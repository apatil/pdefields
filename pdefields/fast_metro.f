      subroutine gmrfmetro (ind,dat,ptr,nnz,nx,x,lp,
     & diag, acc, norms, lv, nv)
!f2py intent(hide) nnz, nx, nv
!f2py intent(inplace) lp, x
      implicit none
      double precision x(nx), dat(nnz), lp(nx)
      double precision diag(nx), acc(nx), norms(nx), mc(nx)
      double precision xp, lpp, lv(nx,nv)
      integer ind(nnz), ptr(nx+1), nx, nnz, i, j, nv
!       x and lp are the current state and log-probability values.
!       They will both be overwritten IN-PLACE.
!
!       ind, dat, ptr are the sparse precision matrix.
!       diag is its diagonal.
!       norms is a vector of N(0,diag(Q)^{-1}) proposal deviates.
!       acc is a vector of Uniform(0,1) acceptance deviates.
!       lv are the other vertex-specific variables used in computing the likelihood.
!       They should be stored as an (nx, nv) array, where nv is the
!       number of variables.
      do i=1,nx
!         mc is the conditional mean of x(i),
!         (Q(i,i) x(i) - Q(i,:) x) / Q(i,i)
          mc(i) = diag(i)*x(i)
          do j=ptr(i)+1,ptr(i+1)
              mc(i) = mc(i) - dat(j)*x(ind(j)+1)
          end do
          mc(i) = mc(i)/diag(i)
          
!         Propose from the conditional prior.
          xp = norms(i)+mc(i)
          
!         This is a template parameter that gets substituted for at runtime.
!         It should contain Fortran code that computes the likelihood, in terms of
!         xp, i, lv and lp.
          {LPF}
          
!         Accept or reject.
          if ((lpp-lp(i)).GT.dlog(acc(i))) then
              lp(i)=lpp
              x(i)=xp
          end if
          
      end do
      
      return
      end
  